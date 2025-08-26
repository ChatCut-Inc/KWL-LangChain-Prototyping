#!/usr/bin/env python3
"""
FastAPI Server for Transcript Analysis
Bridges ChatCut video editor with LangGraph transcript analysis agent
"""

import os
import time
import asyncio
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import json
from dotenv import load_dotenv

# Import your existing LangGraph setup
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
from tools import browse_transcripts as browse_func, read_transcript_text as read_text_func, read_transcript_json as read_json_func, search_transcript_content as search_func

load_dotenv()

# Wrap functions as proper LangChain tools (same as main_verbose.py)
@tool
def browse_transcripts() -> str:
    """List all available transcripts with metadata. Shows title, duration, and speakers 
    for each transcript file."""
    print("browse_transcripts")  # Clean log output
    
    # Use ChatCut transcript data instead of local files
    if not current_transcripts:
        return "No transcript files found in ChatCut media panel. Please add transcripts to your project and enable 'Send to AI'."
    
    result = "Available transcript files:\n"
    for i, transcript in enumerate(current_transcripts):
        # Extract metadata from ChatCut transcript format
        media_name = transcript.get('mediaLibraryItemName', f'Transcript {i+1}')
        duration_ms = transcript.get('durationMs', 0)
        duration_s = duration_ms / 1000.0 if duration_ms else 0
        segments = transcript.get('segments', [])
        
        # Count unique speakers
        speakers = set()
        for segment in segments:
            if segment.get('speaker'):
                speakers.add(segment['speaker'])
        
        result += f"📄 {media_name}: ({duration_s:.0f}s, {len(speakers)} speakers)\n"
    
    return result.strip()

@tool  
def read_transcript_text(filename: str, start_time: float = None, end_time: float = None) -> str:
    """Read transcript as formatted text for content review and analysis. 
    Auto-finds files by name/topic. Returns human-readable conversation format 
    with speaker names and approximate timestamps. Best for brainstorming and quotes."""
    print(f"read_transcript_text({filename})")  # Clean log output
    
    # Find transcript in ChatCut data by name matching
    target_transcript = None
    for transcript in current_transcripts:
        media_name = transcript.get('mediaLibraryItemName', '')
        if filename.lower() in media_name.lower() or media_name.lower() in filename.lower():
            target_transcript = transcript
            break
    
    if not target_transcript:
        # List available transcripts for error message
        available_names = [t.get('mediaLibraryItemName', f'Transcript {i+1}') 
                          for i, t in enumerate(current_transcripts)]
        return f"No transcript found matching '{filename}'. Available: {', '.join(available_names)}"
    
    # Convert ChatCut transcript format to readable text
    segments = target_transcript.get('segments', [])
    media_name = target_transcript.get('mediaLibraryItemName', 'Transcript')
    
    # Apply time filtering if specified
    if start_time is not None or end_time is not None:
        filtered_segments = []
        for segment in segments:
            segment_start = segment.get('start', 0) / 1000.0  # Convert ms to seconds
            segment_end = segment.get('end', 0) / 1000.0
            
            # Include segment if it overlaps with requested time range
            if start_time is not None and segment_end < start_time:
                continue
            if end_time is not None and segment_start > end_time:
                continue
                
            filtered_segments.append(segment)
        segments = filtered_segments
    
    if not segments:
        return f"No content found in the specified time range for {media_name}."
    
    # Format as readable conversation
    result = f"=== {media_name} ===\n\n"
    
    current_speaker = None
    for segment in segments:
        speaker = segment.get('speaker', 'Unknown Speaker')
        text = segment.get('text', '').strip()
        start_ms = segment.get('start', 0)
        
        if not text:
            continue
            
        # Convert milliseconds to MM:SS format
        start_seconds = start_ms // 1000
        minutes = start_seconds // 60
        seconds = start_seconds % 60
        timestamp = f"{minutes:02d}:{seconds:02d}"
        
        # Group consecutive segments from same speaker
        if speaker != current_speaker:
            if current_speaker is not None:
                result += "\n\n"  # Add spacing between speakers
            result += f"**{speaker}** [{timestamp}]: "
            current_speaker = speaker
        else:
            result += " "  # Continue same speaker's text
        
        result += text
    
    return result

@tool  
def read_transcript_json(filename: str, start_time: float = None, end_time: float = None) -> str:
    """Read transcript as JSON with word-level timecodes (millisecond precision).
    Returns complete data structure needed for video editing and precise clips.
    Note: Uses more tokens due to complete timing data for every word."""
    print(f"read_transcript_json({filename})")  # Clean log output
    
    # Find transcript in ChatCut data by name matching
    target_transcript = None
    for transcript in current_transcripts:
        media_name = transcript.get('mediaLibraryItemName', '')
        if filename.lower() in media_name.lower() or media_name.lower() in filename.lower():
            target_transcript = transcript
            break
    
    if not target_transcript:
        # List available transcripts for error message
        available_names = [t.get('mediaLibraryItemName', f'Transcript {i+1}') 
                          for i, t in enumerate(current_transcripts)]
        return f"No transcript found matching '{filename}'. Available: {', '.join(available_names)}"
    
    # Apply time filtering if specified
    filtered_transcript = dict(target_transcript)  # Copy transcript data
    
    if start_time is not None or end_time is not None:
        # Filter segments by time range
        segments = filtered_transcript.get('segments', [])
        filtered_segments = []
        
        for segment in segments:
            segment_start = segment.get('start', 0) / 1000.0  # Convert ms to seconds
            segment_end = segment.get('end', 0) / 1000.0
            
            # Include segment if it overlaps with requested time range
            if start_time is not None and segment_end < start_time:
                continue
            if end_time is not None and segment_start > end_time:
                continue
                
            filtered_segments.append(segment)
        
        filtered_transcript['segments'] = filtered_segments
        filtered_transcript['_time_filter'] = {
            'start_time': start_time,
            'end_time': end_time,
            'original_segments': len(segments),
            'filtered_segments': len(filtered_segments)
        }
    
    import json
    return json.dumps(filtered_transcript, indent=2)

@tool
def search_transcript_content(query: str, limit: int = 10) -> str:
    """Search for specific words or phrases across all transcripts using fast BM25 keyword search.
    Finds exact lines containing the search terms without reading every file.
    Use when user asks to find specific content like 'find all mentions of climate change'."""
    print(f"search_transcript_content('{query}')")  # Clean log output
    
    if not current_transcripts:
        return "No transcripts available to search."
    
    # Search across all ChatCut transcript data
    search_results = []
    query_lower = query.lower()
    
    for transcript in current_transcripts:
        media_name = transcript.get('mediaLibraryItemName', 'Unknown')
        segments = transcript.get('segments', [])
        
        for segment in segments:
            text = segment.get('text', '').strip()
            if not text:
                continue
                
            # Simple case-insensitive search
            if query_lower in text.lower():
                start_ms = segment.get('start', 0)
                start_seconds = start_ms // 1000
                minutes = start_seconds // 60
                seconds = start_seconds % 60
                timestamp = f"{minutes:02d}:{seconds:02d}"
                
                speaker = segment.get('speaker', 'Unknown Speaker')
                
                search_results.append({
                    'media_name': media_name,
                    'timestamp': timestamp,
                    'speaker': speaker,
                    'text': text,
                    'start_ms': start_ms
                })
    
    if not search_results:
        return f"No matches found for '{query}' in any transcript."
    
    # Sort by media name then timestamp
    search_results.sort(key=lambda x: (x['media_name'], x['start_ms']))
    
    # Limit results
    if len(search_results) > limit:
        search_results = search_results[:limit]
    
    # Format results
    result = f"Found {len(search_results)} matches for '{query}':\n\n"
    
    current_media = None
    for match in search_results:
        if match['media_name'] != current_media:
            if current_media is not None:
                result += "\n"
            result += f"=== {match['media_name']} ===\n"
            current_media = match['media_name']
        
        result += f"[{match['timestamp']}] **{match['speaker']}**: {match['text']}\n"
    
    if len(search_results) >= limit:
        result += f"\n... (showing first {limit} results)"
    
    return result

# Initialize FastAPI app
app = FastAPI(
    title="Transcript Analysis API", 
    description="LangGraph agent for analyzing video transcripts",
    version="1.0.0"
)

# Enable CORS for ChatCut frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://localhost:8080"],  # Added port 8080 for ChatCut
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class ChatRequest(BaseModel):
    message: str
    context: dict = {}
    transcripts: list = []

class ChatResponse(BaseModel):
    text: str
    status: str
    tokens: int = 0
    cost: float = 0.0
    tool_calls: list = []  # Track tool calls for UI display

# Global agent (initialized on startup)
agent = None
memory = None

# Global transcript data storage
current_transcripts = []

@app.on_event("startup")
async def startup_event():
    """Initialize the LangGraph agent on server startup"""
    global agent, memory
    
    print("🎬 Starting Transcript Analysis API Server")
    print("=" * 50)
    
    # Setup Gemini model (same as main_verbose.py)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    model = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=1.0,  # Google's 2025 recommendation for better reasoning
        max_retries=2     # Add retry logic for better reliability
    )
    
    tools = [browse_transcripts, read_transcript_text, read_transcript_json, search_transcript_content]
    memory = MemorySaver()
    
    agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=memory
    )
    
    print("🧠 Agent ready! Available endpoints:")
    print("📡 POST /chat - Send messages for transcript analysis") 
    print("📋 GET /transcripts - List available transcripts")
    print(f"📊 LangSmith Project: {os.getenv('LANGSMITH_PROJECT', 'transcript-analyzer')}")
    print(f"🔗 View traces at: https://smith.langchain.com/")
    print("🎯 Ready to serve ChatCut video editor!")

@app.post("/chat/stream")
async def analyze_transcript_stream(request: ChatRequest):
    """
    Streaming endpoint for real-time LangGraph agent events
    Returns Server-Sent Events (SSE) stream
    """
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    async def event_stream():
        try:
            print(f"\n💬 Streaming: {request.message}")
            start_time = time.time()
            
            # Store transcript data globally for tools to access
            global current_transcripts
            current_transcripts = request.transcripts
            print(f"📊 Received {len(current_transcripts)} transcripts from ChatCut")
            
            # Use persistent thread for web sessions
            thread_config = {"configurable": {"thread_id": "chatcut-web-session"}}
            
            full_response = ""
            total_tokens = 0
            
            # Stream events from LangGraph agent using astream_events for tool detection
            async for event in agent.astream_events({
                "messages": [{"role": "user", "content": request.message}]
            }, config=thread_config, version="v2"):
                
                if not isinstance(event, dict):
                    continue
                    
                event_type = event.get("event")
                event_name = event.get("name", "")
                event_data = event.get("data", {})
                
                # Send tool immediately when it completes - chronological order
                if event_type == "on_tool_end":
                    # Extract filename from tool input if available
                    tool_input = event_data.get("input", {})
                    filename = None
                    
                    if isinstance(tool_input, dict) and tool_input.get("filename"):
                        filename = tool_input["filename"]
                    elif isinstance(tool_input, str) and "filename" in tool_input:
                        # Handle string input that might contain filename
                        import re
                        match = re.search(r'filename[\'"]?\s*[:=]\s*[\'"]?([^\'",\s}]+)', tool_input)
                        if match:
                            filename = match.group(1)
                    
                    # Send tool immediately in chronological order
                    tool_data = [{
                        "toolName": event_name,
                        "filename": filename,
                        "status": "completed"
                    }]
                    tool_calls_json = json.dumps(tool_data)
                    tool_content = f"[AGENT_V3_TOOLS]{tool_calls_json}[/AGENT_V3_TOOLS]\n\n"
                    
                    yield f"data: {json.dumps({'type': 'text_chunk', 'content': tool_content})}\n\n"
                    print(f"✅ Tool completed: {event_name}")
                
                # Stream text chunks immediately - no buffering needed
                elif event_type == "on_chat_model_stream":
                    chunk_data = event_data.get("chunk", {})
                    content = getattr(chunk_data, 'content', None) or ""
                    
                    if content:
                        yield f"data: {json.dumps({'type': 'text_chunk', 'content': content})}\n\n"
                        full_response += content
                
                # Extract token usage from chat model completion
                elif event_type == "on_chat_model_end":
                    # Extract token usage if available
                    usage_metadata = event_data.get("usage_metadata")
                    if usage_metadata:
                        total_tokens = usage_metadata.get('input_tokens', 0) + usage_metadata.get('output_tokens', 0)
            
            # Calculate final metrics
            end_time = time.time()
            response_time = end_time - start_time
            estimated_cost = (total_tokens / 1_000_000) * 0.15
            
            print(f"📊 Streamed: {len(full_response)} chars | {total_tokens:,} tokens | {response_time:.1f}s | ${estimated_cost:.4f}")
            
            # Send completion event
            yield f"data: {json.dumps({'type': 'complete', 'tokens': total_tokens, 'cost': estimated_cost, 'response_time': response_time})}\n\n"
            
        except Exception as e:
            print(f"❌ Streaming Error: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "*",
        }
    )

@app.post("/chat", response_model=ChatResponse)
async def analyze_transcript(request: ChatRequest) -> ChatResponse:
    """
    Main endpoint for transcript analysis
    Receives user message and returns LangGraph agent response
    """
    if not agent:
        raise HTTPException(status_code=500, detail="Agent not initialized")
    
    try:
        print(f"\n💬 Received: {request.message}")
        start_time = time.time()
        
        # Store transcript data globally for tools to access
        global current_transcripts
        current_transcripts = request.transcripts
        print(f"📊 Received {len(current_transcripts)} transcripts from ChatCut")
        
        # Use persistent thread for web sessions - maintains memory across requests
        thread_config = {"configurable": {"thread_id": "chatcut-web-session"}}
        
        # Stream the response from your LangGraph agent
        full_response = ""
        total_tokens = 0
        tool_calls = []
        
        
        async for event in agent.astream_events({
            "messages": [{"role": "user", "content": request.message}]
        }, config=thread_config, version="v2"):
            
            # Handle LangGraph event structure (same logic as main_verbose.py)
            if isinstance(event, dict):
                # Collect tool calls from the functions for UI display
                # (Tool calls are already printed in @tool decorators)
                
                if 'agent' in event and 'messages' in event['agent']:
                    messages = event['agent']['messages']
                    if messages:
                        latest_msg = messages[-1]
                        if hasattr(latest_msg, 'content') and latest_msg.content:
                            full_response = latest_msg.content
                            
                            # Extract token usage if available
                            if hasattr(latest_msg, 'usage_metadata'):
                                usage = latest_msg.usage_metadata
                                total_tokens = usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
        
        
        # Calculate metrics
        end_time = time.time()
        response_time = end_time - start_time
        
        # Estimate cost (Gemini 2.0 Flash pricing)
        estimated_cost = (total_tokens / 1_000_000) * 0.15
        
        print(f"📊 Response: {len(full_response)} chars | {total_tokens:,} tokens | {response_time:.1f}s | ${estimated_cost:.4f}")
        
        return ChatResponse(
            text=full_response or "I apologize, but I couldn't generate a response. Please try again.",
            status="success",
            tokens=total_tokens,
            cost=estimated_cost,
            tool_calls=[]  # Clean API - no UI formatting
        )
        
    except Exception as e:
        print(f"❌ Error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/transcripts")
async def get_available_transcripts() -> Dict[str, Any]:
    """
    Optional endpoint to let ChatCut know what transcripts are available
    """
    try:
        transcript_list = browse_func()
        return {
            "transcripts": transcript_list,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list transcripts: {str(e)}")

@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Health check endpoint"""
    return {"status": "healthy", "service": "transcript-analysis-api"}

@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint with API information"""
    return {
        "message": "Transcript Analysis API",
        "description": "LangGraph agent for analyzing video transcripts",
        "endpoints": {
            "POST /chat": "Analyze transcripts with natural language (batch response)",
            "POST /chat/stream": "Analyze transcripts with real-time streaming (SSE)",
            "GET /transcripts": "List available transcript files", 
            "GET /health": "Service health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api_server:app", 
        host="0.0.0.0", 
        port=9000, 
        reload=True,
        log_level="info"
    )