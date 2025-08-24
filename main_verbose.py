#!/usr/bin/env python3
"""
Video Transcript Analyzer - With visible tool calls like Lovable
"""

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain.tools import tool
from tools.transcript_tools import list_transcripts as list_func, read_transcript as read_func

load_dotenv()

# Wrap functions as proper LangChain tools
@tool
def list_transcripts() -> str:
    """List all available transcript files with metadata. Use when user asks what transcripts are available."""
    print("🗂️ list_transcripts")  # Show tool call like Lovable
    return list_func()

@tool  
def read_transcript(filename: str, start_time: float = None, end_time: float = None) -> str:
    """Read specific transcript file content. Use when user asks about transcript content or wants to know what someone said."""
    print(f"🗂️ read_transcript({filename})")  # Show tool call like Lovable
    return read_func(filename, start_time, end_time)

def main():
    print("🎬 Video Transcript Analyzer (Verbose Mode)")
    print("=" * 50)
    
    # Setup Gemini model
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    model = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.1  # Very low temperature for consistent tool usage
    )
    
    tools = [list_transcripts, read_transcript]
    memory = MemorySaver()
    
    agent = create_react_agent(
        model=model,
        tools=tools,
        checkpointer=memory
    )
    
    print("🧠 Agent ready! I will show tool calls like Lovable.")
    print("Try: 'What transcripts do I have?' or 'Tell me about Jesse Katz'")
    print("Type 'quit' to exit.\n")
    
    thread_config = {"configurable": {"thread_id": "verbose-session"}}
    
    # Main interaction loop
    while True:
        user_input = input("🎬 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("👋 Goodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            print()  # Add space before tool calls
            response = agent.invoke({
                "messages": [{"role": "user", "content": user_input}]
            }, config=thread_config)
            
            print(f"\n🤖 Agent: {response['messages'][-1].content}\n")
        
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()