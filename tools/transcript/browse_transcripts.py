"""
Browse all available transcript files with metadata.
"""
import json
import os
import glob


def browse_transcripts() -> str:
    """List all available transcripts with metadata. Shows title, duration, and speakers 
    for each transcript file."""
    transcript_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'transcripts')
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