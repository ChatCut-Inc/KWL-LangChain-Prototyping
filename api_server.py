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
from pydantic import BaseModel
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
    tool_call_display = "🗂️ browse_transcripts"
    print(tool_call_display)  # Show tool call like Lovable
    # Store for UI display
    if hasattr(browse_transcripts, '_current_tool_calls'):
        browse_transcripts._current_tool_calls.append(tool_call_display)
    return browse_func()

@tool  
def read_transcript_text(filename: str, start_time: float = None, end_time: float = None) -> str:
    """Read transcript as formatted text for content review and analysis. 
    Auto-finds files by name/topic. Returns human-readable conversation format 
    with speaker names and approximate timestamps. Best for brainstorming and quotes."""
    # Clean up filename for display
    clean_name = filename.replace('.json', '').replace('_', ' ').replace('-', ' ')
    tool_call_display = f"🗂️ read_transcript_text ({clean_name})"
    print(f"🗂️ read_transcript_text({filename})")  # Show full filename in terminal
    # Store for UI display
    if hasattr(read_transcript_text, '_current_tool_calls'):
        read_transcript_text._current_tool_calls.append(tool_call_display)
    return read_text_func(filename, start_time, end_time)

@tool  
def read_transcript_json(filename: str, start_time: float = None, end_time: float = None) -> str:
    """Read transcript as JSON with word-level timecodes (millisecond precision).
    Returns complete data structure needed for video editing and precise clips.
    Note: Uses more tokens due to complete timing data for every word."""
    # Clean up filename for display  
    clean_name = filename.replace('.json', '').replace('_', ' ').replace('-', ' ')
    tool_call_display = f"🔬 read_transcript_json ({clean_name})"
    print(f"🔬 read_transcript_json({filename})")  # Different icon for experimental
    # Store for UI display
    if hasattr(read_transcript_json, '_current_tool_calls'):
        read_transcript_json._current_tool_calls.append(tool_call_display)
    return read_json_func(filename, start_time, end_time)

@tool
def search_transcript_content(query: str, limit: int = 10) -> str:
    """Search for specific words or phrases across all transcripts using fast BM25 keyword search.
    Finds exact lines containing the search terms without reading every file.
    Use when user asks to find specific content like 'find all mentions of climate change'."""
    tool_call_display = f"🔍 search_transcript_content ('{query}')"
    print(f"🔍 search_transcript_content('{query}')")
    # Store for UI display
    if hasattr(search_transcript_content, '_current_tool_calls'):
        search_transcript_content._current_tool_calls.append(tool_call_display)
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
        model="gemini-2.5-flash",
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
        
        # Initialize tool call collection
        browse_transcripts._current_tool_calls = []
        read_transcript_text._current_tool_calls = []
        read_transcript_json._current_tool_calls = []
        search_transcript_content._current_tool_calls = []
        
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
        
        # Collect all tool calls from the functions
        all_tool_calls = []
        all_tool_calls.extend(browse_transcripts._current_tool_calls)
        all_tool_calls.extend(read_transcript_text._current_tool_calls)
        all_tool_calls.extend(read_transcript_json._current_tool_calls)
        all_tool_calls.extend(search_transcript_content._current_tool_calls)
        
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
            tool_calls=all_tool_calls
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
            "POST /chat": "Analyze transcripts with natural language",
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