#!/usr/bin/env python3
"""
Video Transcript Analyzer - With visible tool calls like Lovable
"""

import os
import time
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
    print("🎬 Video Transcript Analyzer (Verbose Mode + Monitoring)")
    print("=" * 50)
    
    # Session tracking
    session_start_time = time.time()
    total_tokens = 0
    total_cost = 0.0
    query_count = 0
    
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
    
    print("🧠 Agent ready! Monitoring enabled with LangSmith.")
    print("💡 I will show: Tool calls + Token usage + Costs")
    print("Try: 'What transcripts do I have?' or 'Tell me about Jesse Katz'")
    print("Type 'quit' to exit.\n")
    
    print(f"📊 LangSmith Project: {os.getenv('LANGSMITH_PROJECT', 'transcript-analyzer')}")
    print(f"🔗 View traces at: https://smith.langchain.com/\n")
    
    thread_config = {"configurable": {"thread_id": "verbose-session"}}
    
    # Main interaction loop
    while True:
        user_input = input("🎬 You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            # Show session summary
            session_duration = time.time() - session_start_time
            print("\n" + "="*60)
            print("📈 SESSION SUMMARY")
            print("="*60)
            print(f"⏱️  Duration: {session_duration/60:.1f} minutes")
            print(f"💬 Total Queries: {query_count}")
            print(f"🔤 Total Tokens: {total_tokens:,}")
            print(f"💰 Total Cost: ~${total_cost:.4f}")
            print(f"📊 Average per Query: {total_tokens/max(query_count,1):,.0f} tokens, ${total_cost/max(query_count,1):.4f}")
            print(f"\n🔗 View detailed traces: https://smith.langchain.com/")
            print("👋 Goodbye!\n")
            break
        
        if not user_input:
            continue
        
        try:
            print()  # Add space before tool calls
            start_time = time.time()
            
            response = agent.invoke({
                "messages": [{"role": "user", "content": user_input}]
            }, config=thread_config)
            
            # Calculate metrics
            end_time = time.time()
            response_time = end_time - start_time
            query_count += 1
            
            # Extract token usage if available
            query_tokens = 0
            response_msg = response['messages'][-1]
            if hasattr(response_msg, 'usage_metadata'):
                usage = response_msg.usage_metadata
                query_tokens = usage.get('input_tokens', 0) + usage.get('output_tokens', 0)
            
            # Estimate cost (Gemini 1.5 Flash pricing: ~$0.075/$1M input, ~$0.30/$1M output)
            # Rough estimate: $0.15/$1M tokens average
            query_cost = (query_tokens / 1_000_000) * 0.15
            total_tokens += query_tokens
            total_cost += query_cost
            
            print(f"\n🤖 Agent: {response_msg.content}")
            
            # Show metrics
            print(f"\n📊 Tokens: {query_tokens:,} | Cost: ~${query_cost:.4f} | Time: {response_time:.1f}s")
            print(f"💰 Session Total: ${total_cost:.4f} ({query_count} queries)\n")
        
        except Exception as e:
            print(f"❌ Error: {e}\n")

if __name__ == "__main__":
    main()