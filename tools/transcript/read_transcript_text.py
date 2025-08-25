"""
Read transcript with processed text output.
Smart discovery with auto-listing for seamless user experience.
"""
import json
import os
from typing import Optional
from .transcript_discovery import discover_transcript_by_name, get_available_transcript_names
from .browse_transcripts import browse_transcripts


def read_transcript_text(filename: str, start_time: Optional[float] = None, end_time: Optional[float] = None) -> str:
    """
    Read transcript as formatted text for content review and analysis. 
    Auto-finds files by name/topic. Returns human-readable conversation format 
    with speaker names and approximate timestamps. Best for brainstorming and quotes.
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
        
        title = data.get('title', 'Unknown Title')
        duration_ms = data.get('duration_ms', 0)
        duration_s = duration_ms / 1000.0
        
        # Build speaker mapping
        speakers_map = {}
        for speaker in data.get('speakers', []):
            speakers_map[speaker.get('id')] = speaker.get('name', 'Unknown Speaker')
        
        # Convert segments to words with time filtering
        words = []
        for segment in data.get('segments', []):
            speaker_id = segment.get('speaker_id')
            speaker_name = speakers_map.get(speaker_id, 'Unknown Speaker')
            segment_words = segment.get('words', [])
            
            for word_data in segment_words:
                if isinstance(word_data, list) and len(word_data) >= 3:
                    # Format: [text, start_ms, end_ms]
                    word_start_s = word_data[1] / 1000.0  # Convert ms to seconds
                    word_end_s = word_data[2] / 1000.0    # Convert ms to seconds
                    
                    # Apply time filtering if specified
                    if start_time is not None and word_end_s < start_time:
                        continue
                    if end_time is not None and word_start_s > end_time:
                        continue
                    
                    word = {
                        'text': word_data[0],
                        'start': word_start_s,
                        'end': word_end_s,
                        'speaker': speaker_name
                    }
                    words.append(word)
        
        if not words:
            if start_time is not None or end_time is not None:
                return f"No words found in the specified time range ({start_time}-{end_time}s)"
            else:
                return f"No word data found in {os.path.basename(file_path)}"
        
        # Build readable transcript
        result = f"📄 **{title}**\n"
        result += f"Duration: {duration_s:.0f}s | Speakers: {', '.join(speakers_map.values())}\n"
        if start_time is not None or end_time is not None:
            result += f"Time Range: {start_time or 'start'}-{end_time or 'end'}s\n"
        result += "\n"
        
        # Group words by speaker and time for readability
        current_speaker = ""
        current_text = ""
        current_start = None
        
        for word in words:
            speaker = word.get('speaker', 'Unknown')
            text = word.get('text', '')
            start = word.get('start', 0)
            
            if speaker != current_speaker:
                # Output previous speaker's text
                if current_text:
                    timestamp = f"[{current_start:.1f}s]"
                    result += f"{timestamp} **{current_speaker}**: {current_text.strip()}\n"
                
                # Start new speaker
                current_speaker = speaker
                current_text = text + " "
                current_start = start
            else:
                current_text += text + " "
        
        # Output final speaker's text
        if current_text:
            timestamp = f"[{current_start:.1f}s]"
            result += f"{timestamp} **{current_speaker}**: {current_text.strip()}\n"
        
        return result
        
    except json.JSONDecodeError:
        return f"Error: {os.path.basename(file_path)} is not a valid JSON file"
    except Exception as e:
        return f"Error reading {os.path.basename(file_path)}: {str(e)}"