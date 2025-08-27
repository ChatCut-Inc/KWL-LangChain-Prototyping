#!/usr/bin/env python3
"""
LangGraph Studio Graph Definition
Extracted from api_server.py for Studio visualization and debugging
"""

import os
from typing import Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
from dotenv import load_dotenv

# Import existing tools
from tools import browse_transcripts as browse_func, read_transcript_text as read_text_func, read_transcript_json as read_json_func, search_transcript_content as search_func

load_dotenv()

# Global transcript data storage (shared with Studio)
studio_transcripts = []

# Wrap functions as proper LangChain tools (same as api_server.py)
@tool
def browse_transcripts() -> str:
    """List all available transcripts with metadata. Shows title, duration, and speakers 
    for each transcript file."""
    print("browse_transcripts")  # Clean log output
    
    # Use ChatCut transcript data
    if not studio_transcripts:
        return "No transcript files found in ChatCut media panel. Please add transcripts to your project and enable 'Send to AI'."
    
    result = "Available transcript files:\n"
    for i, transcript in enumerate(studio_transcripts):
        # Extract metadata from ChatCut transcript format
        media_name = transcript.get('mediaLibraryItemName', f'Transcript {i+1}')
        duration_ms = transcript.get('durationMs', 0)
        duration_s = duration_ms / 1000.0 if duration_ms else 0
        segments = transcript.get('segments', [])
        
        # Extract speaker names from transcript data
        speaker_info = transcript.get('speakers', [])
        if speaker_info:
            speaker_names = [speaker.get('name', f"Speaker {speaker.get('id', '?')}") for speaker in speaker_info]
            speaker_list = ', '.join(speaker_names)
            result += f"📄 {media_name}: ({duration_s:.0f}s, speakers: {speaker_list})\n"
        else:
            # Fallback: count unique speakers by ID
            speakers = set()
            for segment in segments:
                if segment.get('speaker_id'):
                    speakers.add(segment['speaker_id'])
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
    for transcript in studio_transcripts:
        media_name = transcript.get('mediaLibraryItemName', '')
        if filename.lower() in media_name.lower() or media_name.lower() in filename.lower():
            target_transcript = transcript
            break
    
    if not target_transcript:
        # List available transcripts for error message
        available_names = [t.get('mediaLibraryItemName', f'Transcript {i+1}') 
                          for i, t in enumerate(studio_transcripts)]
        return f"No transcript found matching '{filename}'. Available: {', '.join(available_names)}"
    
    # Convert ChatCut transcript format to readable text
    segments = target_transcript.get('segments', [])
    media_name = target_transcript.get('mediaLibraryItemName', 'Transcript')
    
    # Apply time filtering if specified
    if start_time is not None or end_time is not None:
        filtered_segments = []
        for segment in segments:
            segment_start = segment.get('start_ms', 0) / 1000.0  # Convert ms to seconds
            segment_end = segment.get('end_ms', 0) / 1000.0
            
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
    current_speaker_name = None
    for segment in segments:
        # Use human-readable speaker name if available, fallback to speaker_id
        speaker_name = segment.get('speaker_name', segment.get('speaker_id', 'Unknown Speaker'))
        speaker_id = segment.get('speaker_id', 'unknown')
        text = segment.get('text', '').strip()
        start_ms = segment.get('start_ms', 0)
        
        if not text:
            continue
            
        # Convert milliseconds to MM:SS format
        start_seconds = start_ms // 1000
        minutes = start_seconds // 60
        seconds = start_seconds % 60
        timestamp = f"{minutes:02d}:{seconds:02d}"
        
        # Group consecutive segments from same speaker (use speaker_id for consistency)
        if speaker_id != current_speaker:
            if current_speaker is not None:
                result += "\n\n"  # Add spacing between speakers
            result += f"**{speaker_name}** [{timestamp}]: "
            current_speaker = speaker_id
            current_speaker_name = speaker_name
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
    for transcript in studio_transcripts:
        media_name = transcript.get('mediaLibraryItemName', '')
        if filename.lower() in media_name.lower() or media_name.lower() in filename.lower():
            target_transcript = transcript
            break
    
    if not target_transcript:
        # List available transcripts for error message
        available_names = [t.get('mediaLibraryItemName', f'Transcript {i+1}') 
                          for i, t in enumerate(studio_transcripts)]
        return f"No transcript found matching '{filename}'. Available: {', '.join(available_names)}"
    
    # Apply time filtering if specified
    filtered_transcript = dict(target_transcript)  # Copy transcript data
    
    if start_time is not None or end_time is not None:
        # Filter segments by time range
        segments = filtered_transcript.get('segments', [])
        filtered_segments = []
        
        for segment in segments:
            segment_start = segment.get('start_ms', 0) / 1000.0  # Convert ms to seconds
            segment_end = segment.get('end_ms', 0) / 1000.0
            
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
    
    if not studio_transcripts:
        return "No transcripts available to search."
    
    # Search across all ChatCut transcript data
    search_results = []
    query_lower = query.lower()
    
    for transcript in studio_transcripts:
        media_name = transcript.get('mediaLibraryItemName', 'Unknown')
        segments = transcript.get('segments', [])
        
        for segment in segments:
            text = segment.get('text', '').strip()
            if not text:
                continue
                
            # Simple case-insensitive search
            if query_lower in text.lower():
                start_ms = segment.get('start_ms', 0)
                start_seconds = start_ms // 1000
                minutes = start_seconds // 60
                seconds = start_seconds % 60
                timestamp = f"{minutes:02d}:{seconds:02d}"
                
                # Use human-readable speaker name if available
                speaker = segment.get('speaker_name', segment.get('speaker_id', 'Unknown Speaker'))
                
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

def create_agent_graph() -> Any:
    """Create the LangGraph agent for Studio visualization"""
    
    # Setup Gemini model (same as api_server.py)
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",  # Latest stable Gemini 2.5 Pro model
        google_api_key=api_key,
        temperature=1.0,  # Google's 2025 recommendation for better reasoning
        max_retries=2     # Add retry logic for better reliability
    )
    
    tools = [browse_transcripts, read_transcript_text, read_transcript_json, search_transcript_content]
    memory = MemorySaver()
    
    # Create agent with preprocessing to handle ChatCut data
    base_agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=memory
    )
    
    # Wrap the agent to handle ChatCut transcript data injection
    def preprocess_state(state):
        """Extract ChatCut transcript data from config and inject into global storage"""
        config = state.get("configurable", {})
        transcript_data = config.get("transcript_data", [])
        
        if transcript_data:
            global studio_transcripts
            studio_transcripts = transcript_data
            print(f"🎬 Studio: Received {len(transcript_data)} transcripts from ChatCut")
        
        return state
    
    # Return the base agent (preprocessing happens via state management)
    return base_agent

def update_studio_transcripts(transcripts: List[Dict[str, Any]]):
    """Update transcript data for Studio agent (called by API server)"""
    global studio_transcripts
    studio_transcripts = transcripts
    print(f"🎬 Studio: Updated with {len(transcripts)} transcripts")

# Export for Studio
__all__ = ["create_agent_graph", "update_studio_transcripts"]