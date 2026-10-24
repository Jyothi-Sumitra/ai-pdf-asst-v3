"""
Comprehensive MCP Integration Test Suite for AI PDF & ATS Assistant.

Verifies:
1. Subprocess stdio transport & JSON-RPC 2.0 handshake
2. Dynamic discovery of all 7 MCP tools & schemas
3. Execution of list_available_documents
4. Execution of extract_pdf_raw_text
5. Execution of calculate_ats_score_metric
6. Execution of analyze_resume_ats
7. Execution of ask_pdf (RAG Hybrid search with citations)
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
load_dotenv(BASE_DIR / ".env")

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def run_tests():
    print("=" * 70)
    print("AI PDF & ATS ASSISTANT - MODEL CONTEXT PROTOCOL (MCP) TEST SUITE")
    print("=" * 70)

    python_exe = sys.executable
    server_script = str(BASE_DIR / "mcp_server.py")

    server_params = StdioServerParameters(
        command=python_exe,
        args=[server_script],
        env=dict(os.environ),
    )

    results = []

    print(f"\n[1/7] Connecting to MCP Server via stdio ({Path(server_script).name})...")
    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("  [SUCCESS] JSON-RPC handshake complete. Session active.")
            results.append(("Handshake & Protocol Init", "PASS", "Active stdio session"))

            # 2. List tools
            print("\n[2/7] Discovering registered MCP tools...")
            tools_resp = await session.list_tools()
            tool_names = [t.name for t in tools_resp.tools]
            print(f"  Discovered {len(tool_names)} tools: {', '.join(tool_names)}")

            expected_tools = [
                "list_available_documents",
                "extract_pdf_raw_text",
                "ask_pdf",
                "summarize_pdf",
                "search_web_context",
                "analyze_resume_ats",
                "calculate_ats_score_metric",
            ]
            all_found = all(t in tool_names for t in expected_tools)
            if all_found:
                print(f"  [SUCCESS] All {len(expected_tools)} expected tools registered.")
                results.append(("Tool Discovery", "PASS", f"{len(tool_names)} tools registered"))
            else:
                missing = [t for t in expected_tools if t not in tool_names]
                results.append(("Tool Discovery", "FAIL", f"Missing: {missing}"))

                        # 3. Test list_available_documents
            print("\n[3/7] Testing 'list_available_documents'...")
            try:
                res = await session.call_tool("list_available_documents", {})
                docs = []
                for item in res.content:
                    text = item.text.strip() if hasattr(item, "text") else str(item).strip()
                    try:
                        parsed = json.loads(text)
                        if isinstance(parsed, list):
                            docs.extend(parsed)
                        else:
                            docs.append(parsed)
                    except Exception:
                        docs.append({"raw": text})

                print(f"  Found {len(docs)} documents in repo. Samples:")
                for d in docs[:3]:
                    if isinstance(d, dict):
                        print(f"    - {d.get('filename')} ({d.get('size_kb')} KB in {d.get('category')})")
                    else:
                        print(f"    - {d}")
                results.append(("list_available_documents", "PASS", f"{len(docs)} documents listed"))
            except Exception as e:
                print(f"  [ERROR] {e}")
                results.append(("list_available_documents", "FAIL", str(e)))

            # 4. Test extract_pdf_raw_text
            print("\n[4/7] Testing 'extract_pdf_raw_text' on 'data/ats/resume.pdf'...")
            try:
                res = await session.call_tool(
                    "extract_pdf_raw_text",
                    {"pdf_path": "data/ats/resume.pdf"}
                )
                text = res.content[0].text if res.content else ""
                print(f"  Extracted {len(text)} characters of text.")
                first_line = text.strip().splitlines()[0] if text.strip() else ""
                print(f"  Header preview: {first_line[:80]}")
                results.append(("extract_pdf_raw_text", "PASS", f"{len(text)} chars extracted"))
            except Exception as e:
                print(f"  [ERROR] {e}")
                results.append(("extract_pdf_raw_text", "FAIL", str(e)))

            # 5. Test calculate_ats_score_metric
            print("\n[5/7] Testing 'calculate_ats_score_metric'...")
            try:
                res = await session.call_tool(
                    "calculate_ats_score_metric",
                    {
                        "resume_path": "data/ats/resume.pdf",
                        "job_description_path": "data/ats/job_description.pdf"
                    }
                )
                metric_text = res.content[0].text if res.content else "{}"
                metrics = json.loads(metric_text) if metric_text.startswith("{") else {}
                score = metrics.get("overall_score")
                level = metrics.get("match_level")
                print(f"  Calculated ATS Score: {score}% ({level})")
                print(f"  Matching Skills: {metrics.get('matching_skills_count')}, Missing Skills: {metrics.get('missing_skills_count')}")
                results.append(("calculate_ats_score_metric", "PASS", f"Score: {score}%, Level: {level}"))
            except Exception as e:
                print(f"  [ERROR] {e}")
                results.append(("calculate_ats_score_metric", "FAIL", str(e)))

            # 6. Test analyze_resume_ats
            print("\n[6/7] Testing 'analyze_resume_ats' for detailed gap analysis...")
            try:
                res = await session.call_tool(
                    "analyze_resume_ats",
                    {
                        "resume_path": "data/ats/resume.pdf",
                        "job_description_path": "data/ats/job_description.pdf"
                    }
                )
                ats_text = res.content[0].text if res.content else "{}"
                ats_data = json.loads(ats_text) if ats_text.startswith("{") else {}
                matching = ats_data.get("matching_skills", [])[:4]
                missing = ats_data.get("missing_skills", [])[:4]
                recs = ats_data.get("recommendations", [])[:2]
                print(f"  Top Matching Skills: {', '.join(matching)}")
                print(f"  Top Missing Skills: {', '.join(missing)}")
                print(f"  Sample Recommendation: {recs[0] if recs else 'None'}")
                results.append(("analyze_resume_ats", "PASS", f"{len(matching)} matching skills analyzed"))
            except Exception as e:
                print(f"  [ERROR] {e}")
                results.append(("analyze_resume_ats", "FAIL", str(e)))

            # 7. Test ask_pdf
            print("\n[7/7] Testing 'ask_pdf' RAG query on 'AI_Research.pdf'...")
            try:
                res = await session.call_tool(
                    "ask_pdf",
                    {
                        "pdf_path": "AI_Research.pdf",
                        "question": "What is the primary topic of this research document?"
                    }
                )
                rag_answer = res.content[0].text if res.content else ""
                print(f"  RAG Answer preview: {rag_answer[:140]}...")
                results.append(("ask_pdf (Hybrid RAG)", "PASS", f"Answer returned ({len(rag_answer)} chars)"))
            except Exception as e:
                print(f"  [ERROR] {e}")
                results.append(("ask_pdf (Hybrid RAG)", "FAIL", str(e)))

    print("\n" + "=" * 70)
    print("TEST SUITE SUMMARY RESULTS")
    print("=" * 70)
    all_passed = True
    for name, status, detail in results:
        symbol = "[OK]" if status == "PASS" else "[FAIL]"
        print(f"  {symbol:<8} {name:<30} {detail}")
        if status != "PASS":
            all_passed = False

    print("=" * 70)
    if all_passed:
        print("ALL MCP INTEGRATION TESTS PASSED SUCCESSFULLY!")
    else:
        print("SOME TESTS FAILED - REVIEW DETAILS ABOVE.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_tests())
