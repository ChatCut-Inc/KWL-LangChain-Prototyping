import json
import os
import glob
from typing import Optional

def list_transcripts() -> str:
    """Show all available transcript files with metadata (title, duration, speakers)."""
    transcript_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'transcripts')
    json_files = glob.glob(os.path.join(transcript_dir, '*.json'))
    
    if not json_files:
        return "No transcript files found in the transcripts folder."
    
    result = "Available transcript files:\n"
    for file_path in json_files:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                title = data.get('title', 'No title')
                duration_ms = data.get('duration_ms', 0)
                duration_s = duration_ms / 1000.0
                speakers = data.get('speakers', [])
                result += f"📄 {filename}: {title} ({duration_s:.0f}s, {len(speakers)} speakers)\n"
        except Exception as e:
            result += f"📄 {filename}: (Error reading file)\n"
    
    return result.strip()

def read_transcript(filename: str, start_time: Optional[float] = None, end_time: Optional[float] = None) -> str:
    """Read transcript content by name or topic. Automatically finds matching files using smart search. Use when user asks about any transcript content (e.g., 'chuck', 'gang reporter', 'pharma', or exact filenames)."""
    transcript_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'transcripts')
    file_path = os.path.join(transcript_dir, filename)
    
    # If exact filename doesn't exist, try smart matching
    if not os.path.exists(file_path):
        # Get list of available transcripts for smart matching
        json_files = glob.glob(os.path.join(transcript_dir, '*.json'))
        
        if not json_files:
            return "No transcript files found."
            
        # Smart matching logic
        matched_file = None
        filename_lower = filename.lower()
        
        # Try exact match first (case insensitive)
        for file_path_candidate in json_files:
            if filename_lower == os.path.basename(file_path_candidate).lower():
                matched_file = file_path_candidate
                break
        
        # Try partial matching on filename and content
        if not matched_file:
            best_match = None
            best_score = 0
            
            for file_path_candidate in json_files:
                basename = os.path.basename(file_path_candidate).lower()
                
                # Check if search term is in filename
                score = 0
                if filename_lower in basename:
                    score += 10
                
                # Check specific keywords
                if 'chuck' in filename_lower or 'palahniuk' in filename_lower:
                    if 'chuck' in basename or 'palahniuk' in basename:
                        score += 20
                elif 'jesse' in filename_lower or 'katz' in filename_lower or 'gang' in filename_lower:
                    if 'jesse' in basename or 'katz' in basename or 'gang' in basename:
                        score += 20
                elif 'ron' in filename_lower or 'piana' in filename_lower or 'pharma' in filename_lower or 'greed' in filename_lower:
                    if 'ron' in basename or 'piana' in basename or 'pharma' in basename or 'greed' in basename:
                        score += 20
                
                if score > best_score:
                    best_score = score
                    best_match = file_path_candidate
            
            if best_match and best_score > 0:
                matched_file = best_match
        
        if not matched_file:
            # Show available options
            available_files = [os.path.basename(f) for f in json_files]
            return f"No transcript found matching '{filename}'. Available transcripts: {', '.join(available_files)}"
        
        file_path = matched_file
    
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
                return f"No word data found in {filename}"
        
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
        return f"Error: {filename} is not a valid JSON file"
    except Exception as e:
        return f"Error reading {filename}: {str(e)}"