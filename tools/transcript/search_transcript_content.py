"""
Search transcript content using BM25S for fast keyword/phrase search.
Finds specific lines containing search terms without reading all files.
"""
import json
import os
import glob
import re
from typing import List, Dict, Any, Optional
try:
    import bm25s
except ImportError:
    bm25s = None


class TranscriptSearchIndex:
    """BM25S-based search index for transcript content."""
    
    def __init__(self):
        self.index = None
        self.documents = []  # Store document metadata
        self.corpus_segments = []  # Store searchable text segments
        self.is_built = False
    
    def build_index(self, transcript_dir: str) -> str:
        """Build search index from all transcript files."""
        if bm25s is None:
            return "BM25S library not installed. Run: pip install bm25s"
        
        json_files = glob.glob(os.path.join(transcript_dir, '*.json'))
        
        if not json_files:
            return "No transcript files found to index."
        
        # Prepare corpus for indexing
        corpus_text = []
        self.documents = []
        self.corpus_segments = []
        
        for file_path in json_files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                filename = os.path.basename(file_path)
                title = data.get('title', 'Unknown Title')
                
                # Extract text segments with speaker and timing info
                segments = self._extract_segments(data)
                
                for segment in segments:
                    # Create searchable text
                    text = f"{segment['speaker']}: {segment['text']}"
                    corpus_text.append(text)
                    
                    # Store metadata for results
                    self.corpus_segments.append({
                        'file': filename,
                        'title': title,
                        'speaker': segment['speaker'],
                        'text': segment['text'],
                        'start_time': segment['start_time'],
                        'end_time': segment['end_time']
                    })
                
                self.documents.append({
                    'filename': filename,
                    'title': title,
                    'segments_count': len(segments)
                })
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
        
        if not corpus_text:
            return "No searchable content found in transcript files."
        
        # Build BM25S index
        try:
            # Tokenize corpus
            corpus_tokens = bm25s.tokenize(corpus_text, stopwords="en")
            
            # Create and build index
            self.index = bm25s.BM25()
            self.index.index(corpus_tokens)
            self.is_built = True
            
            return f"Successfully indexed {len(corpus_text)} segments from {len(self.documents)} transcript files."
            
        except Exception as e:
            return f"Error building search index: {e}"
    
    def search(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search for query in transcript content."""
        if not self.is_built or self.index is None:
            return []
        
        try:
            # Tokenize query
            query_tokens = bm25s.tokenize(query, stopwords="en")
            
            # Perform search
            docs, scores = self.index.retrieve(query_tokens, k=min(limit, len(self.corpus_segments)))
            
            # Format results
            results = []
            for i, (doc_idx, score) in enumerate(zip(docs[0], scores[0])):
                if score > 0:  # Only include relevant results
                    segment = self.corpus_segments[doc_idx]
                    results.append({
                        'relevance_score': float(score),
                        'file': segment['file'],
                        'title': segment['title'],
                        'speaker': segment['speaker'],
                        'text': segment['text'],
                        'start_time': segment['start_time'],
                        'end_time': segment['end_time'],
                        'timestamp': f"[{segment['start_time']:.1f}s]"
                    })
            
            return results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def _extract_segments(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract text segments from transcript data."""
        segments = []
        
        # Build speaker mapping
        speakers_map = {}
        for speaker in data.get('speakers', []):
            speakers_map[speaker.get('id')] = speaker.get('name', 'Unknown Speaker')
        
        # Process segments
        for segment in data.get('segments', []):
            speaker_id = segment.get('speaker_id')
            speaker_name = speakers_map.get(speaker_id, 'Unknown Speaker')
            segment_words = segment.get('words', [])
            
            if not segment_words:
                continue
            
            # Combine words into text with timing
            words_text = []
            start_time = None
            end_time = None
            
            for word_data in segment_words:
                if isinstance(word_data, list) and len(word_data) >= 3:
                    word_text = word_data[0]
                    word_start_ms = word_data[1]
                    word_end_ms = word_data[2]
                    
                    words_text.append(word_text)
                    
                    # Track segment timing
                    if start_time is None:
                        start_time = word_start_ms / 1000.0
                    end_time = word_end_ms / 1000.0
            
            if words_text and start_time is not None:
                segments.append({
                    'speaker': speaker_name,
                    'text': ' '.join(words_text),
                    'start_time': start_time,
                    'end_time': end_time
                })
        
        return segments


# Global search index instance
_search_index = TranscriptSearchIndex()


def search_transcript_content(query: str, limit: int = 10, rebuild_index: bool = False) -> str:
    """
    Search for specific content across all transcript files using BM25.
    Fast keyword/phrase search that returns matching lines with context.
    
    Args:
        query: Search query (e.g., "climate change", "artificial intelligence")
        limit: Maximum number of results to return (default: 10)
        rebuild_index: Whether to rebuild the search index (default: False)
    
    Returns:
        Formatted search results with timestamps and speaker information
    """
    # Find transcript directory - try multiple paths
    potential_paths = [
        os.path.join(os.getcwd(), 'transcripts'),  # From project root
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'transcripts'),  # Relative to tools
        '/Users/kaiwenli/langgraph_project/transcripts'  # Absolute path as fallback
    ]
    
    transcript_dir = None
    for path in potential_paths:
        if os.path.exists(path) and any(f.endswith('.json') for f in os.listdir(path)):
            transcript_dir = path
            break
    
    if not transcript_dir:
        return "Transcript directory not found. Please ensure transcripts are available."
    
    # Build index if needed
    if not _search_index.is_built or rebuild_index:
        result = _search_index.build_index(transcript_dir)
        if "Error" in result or "not installed" in result or "No transcript files" in result:
            return result
    
    # Perform search
    results = _search_index.search(query, limit)
    
    if not results:
        return f"No results found for '{query}'. Try different keywords or check if transcripts are available."
    
    # Format results
    formatted_results = f"🔍 **Search Results for '{query}'**\n"
    formatted_results += f"Found {len(results)} matches:\n\n"
    
    for i, result in enumerate(results, 1):
        formatted_results += f"**{i}. {result['title']}** ({result['file']})\n"
        formatted_results += f"{result['timestamp']} **{result['speaker']}**: {result['text']}\n"
        formatted_results += f"_Relevance: {result['relevance_score']:.3f}_\n\n"
    
    return formatted_results.strip()