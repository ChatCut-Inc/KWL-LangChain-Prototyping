"""
Read transcript with raw JSON output for AI processing.
Experimental approach - lets AI handle formatting directly.
"""
import json
import os
from typing import Optional
from .transcript_discovery import discover_transcript_by_name, get_available_transcript_names
from .browse_transcripts import browse_transcripts


def read_transcript_json(filename: str, start_time: Optional[float] = None, end_time: Optional[float] = None) -> str:
    """
    Read transcript as JSON with word-level timecodes (millisecond precision).
    Returns complete data structure needed for video editing and precise clips.
    Note: Uses more tokens due to complete timing data for every word.
    """
    transcript_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'transcripts')
    
    # Use smart discovery to find the transcript
    file_path = discover_transcript_by_name(filename, transcript_dir, browse_transcripts)
    
    if not file_path:
        # Get available transcripts for error message
        transcript_listing = browse_transcripts()
        lines = transcript_listing.split('\n')
        available_transcripts = {}
        for line in lines[1:]:
            if line.strip() and '📄' in line:
                parts = line.split(':', 1)
                if len(parts) >= 2:
                    filename_part = parts[0].replace('📄', '').strip()
                    available_transcripts[filename_part] = ""
        
        available_names = get_available_transcript_names(available_transcripts)
        return f"No transcript found matching '{filename}'. Available: {available_names}"
    
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Add time filtering instructions if specified
        if start_time or end_time:
            data['_processing_instructions'] = {
                'filter_time_range': True,
                'start_time_seconds': start_time,
                'end_time_seconds': end_time
            }
        
        # Add helpful context for the AI
        data['_ai_instructions'] = {
            'task': 'Process this transcript data and provide a readable summary',
            'format_preference': 'Include speaker names, key topics, and timestamps when relevant',
            'source_file': os.path.basename(file_path)
        }
        
        return json.dumps(data, indent=2)
        
    except json.JSONDecodeError:
        return f"Error: {os.path.basename(file_path)} is not a valid JSON file"
    except Exception as e:
        return f"Error reading {os.path.basename(file_path)}: {str(e)}"