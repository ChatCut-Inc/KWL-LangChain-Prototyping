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
    return browse_func()

@tool  
def read_transcript_text(filename: str, start_time: float = None, end_time: float = None) -> str:
    """Read transcript as formatted text for content review and analysis. 
    Auto-finds files by name/topic. Returns human-readable conversation format 
    with speaker names and approximate timestamps. Best for brainstorming and quotes."""
    print(f"read_transcript_text({filename})")  # Clean log output
    return read_text_func(filename, start_time, end_time)

@tool  
def read_transcript_json(filename: str, start_time: float = None, end_time: float = None) -> str:
    """Read transcript as JSON with word-level timecodes (millisecond precision).
    Returns complete data structure needed for video editing and precise clips.
    Note: Uses more tokens due to complete timing data for every word."""
    print(f"read_transcript_json({filename})")  # Clean log output
    return read_json_func(filename, start_time, end_time)

@tool
def search_transcript_content(query: str, limit: int = 10) -> str:
    """Search for specific words or phrases across all transcripts using fast BM25 keyword search.
    Finds exact lines containing the search terms without reading every file.
    Use when user asks to find specific content like 'find all mentions of climate change'."""
    print(f"search_transcript_content('{query}')")  # Clean log output
    return search_func(query, limit)

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

class ChatResponse(BaseModel):
    text: str
    status: str
    tokens: int = 0
    cost: float = 0.0
    tool_calls: list = []  # Track tool calls for UI display

# Global agent (initialized on startup)
agent = None
memory = None

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
            
            # Use persistent thread for web sessions
            thread_config = {"configurable": {"thread_id": "chatcut-web-session"}}
            
            full_response = ""
            total_tokens = 0
            
            # Stream events from LangGraph agent
            async for event in agent.astream({
                "messages": [{"role": "user", "content": request.message}]
            }, config=thread_config):
                
                if isinstance(event, dict):
                    # Handle different LangGraph event types
                    for node_name, node_data in event.items():
                        
                        # Agent messages (includes tool calls)
                        if node_name == "agent" and isinstance(node_data, dict):
                            if "messages" in node_data:
                                messages = node_data["messages"]
                                if messages:
                                    latest_msg = messages[-1]
                                    
                                    # Check for tool calls in agent messages
                                    if hasattr(latest_msg, 'tool_calls') and latest_msg.tool_calls:
                                        for tool_call in latest_msg.tool_calls:
                                            tool_name = tool_call.get('name', 'unknown_tool')
                                            tool_args = tool_call.get('args', {})
                                            yield f"data: {json.dumps({'type': 'tool_start', 'tool': tool_name, 'args': tool_args})}\n\n"
                                    
                                    # Handle text content
                                    if hasattr(latest_msg, 'content') and latest_msg.content:
                                        # Check if this is new content to stream
                                        new_content = latest_msg.content
                                        if new_content != full_response:
                                            if full_response and new_content.startswith(full_response):
                                                # Stream just the new part
                                                new_part = new_content[len(full_response):]
                                                yield f"data: {json.dumps({'type': 'text_chunk', 'content': new_part})}\n\n"
                                            else:
                                                # Stream full content if it's different
                                                yield f"data: {json.dumps({'type': 'text_chunk', 'content': new_content})}\n\n"
                                            full_response = new_content
                                        
                                        # Extract token usage if available
                                        if hasattr(latest_msg, 'usage_metadata'):
                                            usage = latest_msg.usage_metadata
                                            total_tokens = usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
                        
                        # Tool execution results
                        elif node_name == "tools" and isinstance(node_data, dict):
                            if "messages" in node_data:
                                for message in node_data["messages"]:
                                    if hasattr(message, 'content'):  # Tool result
                                        yield f"data: {json.dumps({'type': 'tool_end', 'tool': 'completed', 'status': 'success'})}\n\n"
            
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
        
        # Use persistent thread for web sessions - maintains memory across requests
        thread_config = {"configurable": {"thread_id": "chatcut-web-session"}}
        
        # Stream the response from your LangGraph agent
        full_response = ""
        total_tokens = 0
        tool_calls = []
        
        
        async for event in agent.astream({
            "messages": [{"role": "user", "content": request.message}]
        }, config=thread_config):
            
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