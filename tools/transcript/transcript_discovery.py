"""
Shared transcript discovery and matching logic.
Generic, scalable approach that works with any transcript content.
"""
import os
import glob
from typing import Dict, Optional, Tuple


def discover_transcript_by_name(filename: str, transcript_dir: str, list_transcripts_func) -> Optional[str]:
    """
    Find transcript file using smart matching with metadata.
    
    Args:
        filename: User's search term (e.g., 'chuck', 'gang reporter', exact filename)
        transcript_dir: Directory containing transcript files
        list_transcripts_func: Function that returns rich transcript listing
        
    Returns:
        Full path to matched transcript file, or None if no match found
    """
    # Try exact path first
    file_path = os.path.join(transcript_dir, filename)
    if os.path.exists(file_path):
        return file_path
    
    # Get rich transcript listing with metadata (titles, speakers, etc.)
    transcript_listing = list_transcripts_func()
    
    if "No transcript files found" in transcript_listing:
        return None
    
    # Parse the transcript listing to extract filenames and metadata
    available_transcripts = _parse_transcript_listing(transcript_listing)
    
    if not available_transcripts:
        return None
    
    # Find best match using generic scoring
    best_match = _find_best_match(filename.lower(), available_transcripts)
    
    if best_match:
        return os.path.join(transcript_dir, best_match)
    
    return None


def _parse_transcript_listing(transcript_listing: str) -> Dict[str, str]:
    """Parse transcript listing to extract filenames and metadata."""
    lines = transcript_listing.split('\n')
    available_transcripts = {}
    
    for line in lines[1:]:  # Skip "Available transcript files:" header
        if line.strip() and '📄' in line:
            # Parse format: "📄 filename.json: Title (duration, speakers)"
            parts = line.split(':', 1)
            if len(parts) >= 2:
                filename_part = parts[0].replace('📄', '').strip()
                metadata_part = parts[1].strip()
                available_transcripts[filename_part] = metadata_part.lower()
    
    return available_transcripts


def _find_best_match(search_term: str, available_transcripts: Dict[str, str]) -> Optional[str]:
    """
    Find best matching transcript using generic scoring algorithm.
    No hardcoded keywords - works with any content.
    """
    best_match = None
    best_score = 0
    
    for filename, metadata in available_transcripts.items():
        # Try exact filename match first (highest priority)
        if search_term == filename.lower():
            return filename
        
        # Generic scoring approach
        score = 0
        filename_lower = filename.lower()
        
        # Filename substring matching
        if search_term in filename_lower:
            score += 10
        
        # Rich metadata matching (titles, descriptions)
        if search_term in metadata:
            score += 15  # Metadata matches are more valuable
        
        # Partial word matching in both filename and metadata
        search_words = search_term.split()
        for word in search_words:
            if len(word) >= 3:  # Only match meaningful words
                if word in filename_lower:
                    score += 5
                if word in metadata:
                    score += 8
        
        if score > best_score:
            best_score = score
            best_match = filename
    
    # Only return match if score is meaningful
    return best_match if best_score >= 5 else None


def get_available_transcript_names(available_transcripts: Dict[str, str]) -> str:
    """Format available transcript names for error messages."""
    return ', '.join(available_transcripts.keys())