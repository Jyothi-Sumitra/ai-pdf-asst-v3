"""
Autonomous MCP Agent for AI PDF & ATS Assistant.

This agent connects to mcp_server.py via stdio JSON-RPC, dynamically
discovers all registered MCP tools, binds them natively to Groq LLM,
and runs an autonomous multi-step reasoning loop to execute document tasks.
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

# Base directory
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

load_dotenv(BASE_DIR / ".env")

# Ensure stdout/stderr utf-8 encoding on Windows to prevent charmap errors
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from src.chatbot import get_llm


AGENT_SYSTEM_PROMPT = """You are an Autonomous AI Document & ATS Intelligence Agent.
You have access to a suite of Model Context Protocol (MCP) tools provided by the AI PDF & ATS Assistant.

Guidelines:
1. When asked about documents, inspect available documents using 'list_available_documents' if the path is unknown.
2. For questions regarding PDF content, use 'ask_pdf' (hybrid RAG with page citations) or 'summarize_pdf'.
3. For resume evaluations against job descriptions, use 'analyze_resume_ats' or 'calculate_ats_score_metric'.
4. For live web grounding or supplementary real-time facts, use 'search_web_context'.
5. Always provide clear, well-structured, professional final responses.
"""


def _mcp_tools_to_openai_specs(tools: list) -> list:
    """Convert MCP tool objects to OpenAI-compatible function specs for LangChain."""
    specs = []
    for tool in tools:
        schema = getattr(tool, "inputSchema", {}) or {"type": "object", "properties": {}}
        specs.append({
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": schema
            }
        })
    return specs


async def run_agent_turn(
    session: ClientSession,
    llm,
    user_query: str,
    max_steps: int = 6
) -> str:
    """Execute multi-step autonomous tool reasoning loop."""
    # 1. Discover tools dynamically from MCP server
    tools_response = await session.list_tools()
    tools = tools_response.tools
    tool_specs = _mcp_tools_to_openai_specs(tools)
    tool_names = [t.name for t in tools]

    llm_with_tools = llm.bind_tools(tool_specs)

    print(f"\n\033[1;36m=== Autonomous MCP Agent Turn ===\033[0m")
    print(f"\033[1;33mTask:\033[0m {user_query}")
    print(f"\033[90mActive MCP Tools ({len(tool_names)}): {', '.join(tool_names)}\033[0m\n")

    conversation = [
        SystemMessage(content=AGENT_SYSTEM_PROMPT),
        HumanMessage(content=user_query)
    ]

    for step in range(1, max_steps + 1):
        print(f"\033[1;34m[Step {step}/{max_steps}] Reasoning with LLM...\033[0m")
        response = llm_with_tools.invoke(conversation)
        conversation.append(response)

        # If LLM did not request any tool calls, this is the final answer
        if not response.tool_calls:
            final_text = response.content.strip()
            print(f"\n\033[1;32m=== Final Response ===\033[0m\n{final_text}\n")
            return final_text

        # Process each tool call requested by the model
        for tool_call in response.tool_calls:
            t_name = tool_call["name"]
            t_args = tool_call["args"]
            t_id = tool_call.get("id", f"call_{step}")

            print(f"\033[1;35m-> Executing MCP Tool:\033[0m \033[4m{t_name}\033[0m")
            print(f"   Args: {json.dumps(t_args, ensure_ascii=False)}")

            try:
                result = await session.call_tool(t_name, t_args)
                chunks = []
                for item in result.content:
                    if hasattr(item, "text"):
                        chunks.append(item.text)
                    else:
                        chunks.append(str(item))
                observation = "\n".join(chunks)
            except Exception as e:
                observation = f"Tool execution failed: {type(e).__name__}: {str(e)}"

            preview = observation[:240] + ("..." if len(observation) > 240 else "")
            print(f"\033[92m   Observation ({len(observation)} chars):\033[0m {preview}\n")

            conversation.append(ToolMessage(content=observation, tool_call_id=t_id))

    # Fallback if loop exceeded
    print("\033[33mWarning: Reached maximum reasoning steps. Synthesizing available findings...\033[0m")
    summary_resp = llm.invoke(conversation + [HumanMessage(content="Summarize your final answer based on the above observations.")])
    print(f"\n\033[1;32m=== Final Response ===\033[0m\n{summary_resp.content.strip()}\n")
    return summary_resp.content.strip()


async def run_client_session(user_query: Optional[str] = None):
    """Start MCP stdio subprocess, initialize session, and execute agent query."""
    python_exe = sys.executable
    server_script = str(BASE_DIR / "mcp_server.py")

    server_params = StdioServerParameters(
        command=python_exe,
        args=[server_script],
        env=dict(os.environ),
    )

    print(f"\033[1;32mStarting stdio connection to MCP Server at: {server_script}\033[0m")

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("\033[1;32mMCP Handshake Complete. Session initialized.\033[0m")

            llm = get_llm()

            if user_query:
                await run_agent_turn(session, llm, user_query)
            else:
                # Interactive CLI mode
                print("\n\033[1;34m=== Interactive MCP Agent Ready (type 'exit' to quit) ===\033[0m")
                while True:
                    try:
                        q = input("\n\033[1;37mAgent Prompt > \033[0m").strip()
                        if not q:
                            continue
                        if q.lower() in ("exit", "quit", "q"):
                            print("Exiting agent. Goodbye!")
                            break
                        await run_agent_turn(session, llm, q)
                    except (KeyboardInterrupt, EOFError):
                        print("\nExiting.")
                        break


def main():
    parser = argparse.ArgumentParser(description="Autonomous MCP Agent for PDF & ATS Assistant")
    parser.add_argument("--query", "-q", type=str, help="Single query for the agent to execute")
    parser.add_argument("--demo", action="store_true", help="Run automated demonstration on ATS and PDF data")
    args = parser.parse_args()

    demo_query = (
        "Calculate the ATS match score between data/ats/resume.pdf and "
        "data/ats/job_description.pdf, then analyze what specific recommendations "
        "the candidate should follow to improve their score."
    )

    query = demo_query if args.demo else args.query
    asyncio.run(run_client_session(query))


if __name__ == "__main__":
    main()
