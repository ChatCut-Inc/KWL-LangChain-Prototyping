# Video Transcript Analyzer

A LangGraph-powered agent that analyzes video transcripts with word-level timing data.

## 📁 Project Structure

```
langgraph_project/
├── main.py                        # Main transcript analyzer agent
├── transcripts/                   # Drop your JSON transcript files here
│   ├── sample_meeting.json        # Sample: team meeting transcript
│   └── interview_ceo.json         # Sample: CEO interview transcript
├── tools/
│   ├── __init__.py
│   └── transcript_tools.py        # Core transcript analysis functions
├── test_transcript_analyzer.py    # Test script for functionality
└── requirements.txt               # Python dependencies
```

## 🚀 How to Use

1. **Run in VSCode**: Press F5 and select "Run Transcript Analyzer"
2. **Add your transcripts**: Drop JSON files in the `transcripts/` folder
3. **Ask questions**: The agent will smartly analyze your transcript data

## 🛠️ Core Tools

- `list_transcripts()` - Shows available transcript files
- `read_transcript()` - Reads specific transcripts with time filtering
- `search_transcripts()` - Searches content across all files
- `get_transcript_summary()` - Generates statistics and summaries

## 💬 Example Questions

- "What transcript files do I have?"
- "Search for 'AI' across all transcripts"
- "Show me what Alice said in the meeting"
- "Give me a summary of the CEO interview"
- "What was discussed in the first 10 seconds?"

## 📊 JSON Format

Your transcript files should follow this structure:
```json
{
  "title": "Video Title",
  "duration": 1800,
  "speakers": ["Speaker1", "Speaker2"],
  "words": [
    {"text": "hello", "start": 1.0, "end": 1.5, "speaker": "Speaker1"},
    {"text": "world", "start": 1.6, "end": 2.0, "speaker": "Speaker1"}
  ]
}
```

## 🧠 Features

- **Memory**: Remembers conversation context
- **Smart Search**: Finds relevant content across files
- **Time-aware**: Uses word-level timing for precise queries
- **Speaker Analysis**: Tracks who said what and when
- **Flexible**: Works with any JSON transcript format