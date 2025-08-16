# Agent logic and graph construction for the local AI research project

from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

from tavily import TavilyClient

import streamlit as st
import os
import traceback

from schemas import ReportState, QueryResult
from prompts import (
    build_queries,
    resume_reasearch,
    build_final_response,
)

from dotenv import load_dotenv

load_dotenv()


# Models
llm = ChatOllama(model="gemma3:4b")
reasoning_llm = ChatOllama(model="llama3.2:3b")


# Logging function
def _log(msg):
    print(msg)
    try:
        st.write(msg)
    except Exception:
        pass


# Nodes
def build_first_queries(
    state: ReportState,
) -> dict:  # This dictionary will change the state
    class QueryList(BaseModel):
        queries: list[str]

    query_llm = llm.with_structured_output(QueryList)

    user_input = state.user_input
    prompt = build_queries.format(user_input=user_input)

    _log(f"[build_first_queries] prompt: {prompt}")

    result = None
    try:
        result = query_llm.invoke(prompt)
        _log(f"[build_first_queries] LLM returned: {result}")
    except Exception as e:
        tb = traceback.format_exc()
        _log(f"[build_first_queries] LLM invoke failed: {e}\n{tb}")

        if user_input:
            fallback = [s.strip() for s in user_input.split("?") if s.strip()]
            if not fallback:
                fallback = [user_input]
            _log(f"[build_first_queries] using fallback queries: {fallback}")
            return {"queries": fallback}
        else:
            return {"queries": []}

    queries = getattr(result, "queries", None)
    if not queries:
        _log("[build_first_queries] no queries produced by LLM, using fallback")
        if user_input:
            return {"queries": [user_input]}
        return {"queries": []}

    return {"queries": result.queries}


def single_search(payload: dict) -> dict:
    query = payload["query"]
    user_input = payload["user_input"]
    _log(f"[single_search] query={query} user_input={user_input}")

    tavily_client = None
    try:
        tavily_client = TavilyClient()
    except Exception as e:
        _log(f"[single_search] Error creating TavilyClient: {e}")

    results = None
    if tavily_client:
        try:
            results = tavily_client.search(
                query, max_results=1, include_raw_content=False
            )
            _log(f"[single_search] tavily.search returned: {results}")
        except Exception as e:
            tb = traceback.format_exc()
            _log(f"[single_search] tavily.search failed: {e}\n{tb}")

    # fallback result if API failed
    if not results or "results" not in results or not results["results"]:
        _log("[single_search] using fallback/dummy search result")
        dummy_url = f"https://example.com/search?q={query.replace(' ', '+')}"
        results = {
            "results": [
                {
                    "title": f"Fallback result for {query}",
                    "url": dummy_url,
                    "content": "",
                    "raw_content": "",
                }
            ]
        }

    url = results["results"][0].get("url")
    url_extracted = {}
    if tavily_client:
        try:
            url_extracted = tavily_client.extract(url)
            _log(f"[single_search] tavily.extract returned: {url_extracted}")
        except Exception as e:
            tb = traceback.format_exc()
            _log(f"[single_search] tavily.extract failed: {e}\n{tb}")
            url_extracted = {}
    else:
        url_extracted = {"results": [{"raw_content": ""}]}

    llm_result = None
    try:
        if len(url_extracted.get("results", [])) > 0:
            raw_content = url_extracted["results"][0].get("raw_content", "")
            prompt = resume_reasearch.format(
                user_input=user_input, search_results=raw_content
            )
            _log(
                f"[single_search] calling llm for resume with prompt length {len(prompt)}"
            )
            llm_result = llm.invoke(prompt)
            _log("[single_search] llm returned for resume")
    except Exception as e:
        tb = traceback.format_exc()
        _log(f"[single_search] llm.invoke failed: {e}\n{tb}")
        llm_result = None

    resume_text = getattr(llm_result, "content", "") if llm_result is not None else ""
    query_result = QueryResult(
        title=results["results"][0].get("title"), url=url, resume=resume_text
    )

    return {"queries_results": [query_result]}


def final_writer(state: ReportState):
    _log(f"[final_writer] building final from {len(state.queries_results)} results")
    search_results = ""
    references = ""
    for i, result in enumerate(state.queries_results):
        search_results += f"[{i + 1}]\n\n"
        search_results += f"Title: {result.title}\n"
        search_results += f"URL: {result.url}\n"
        search_results += f"Content: {result.resume}\n"
        search_results += f"================\n\n"

        references += f"[{i + 1}] - [{result.title}]({result.url})\n"

    prompt = build_final_response.format(
        user_input=state.user_input, search_results=search_results
    )

    llm_result = None
    try:
        llm_result = reasoning_llm.invoke(prompt)
        _log("[final_writer] reasoning_llm returned")
    except Exception as e:
        tb = traceback.format_exc()
        _log(f"[final_writer] reasoning_llm.invoke failed: {e}\n{tb}")
        # fallback: simple aggregation
        llm_result = None

    if llm_result and getattr(llm_result, "content", None):
        final_response = llm_result.content + "\n\n References:\n" + references
    else:
        final_response = (
            "Unable to generate final response from LLM. Here are the aggregated results:\n\n"
            + search_results
            + "\n\nReferences:\n"
            + references
        )

    return {"final_response": final_response}


def collect_results(state: ReportState) -> dict:
    """
    A new node to collect all the research results from the parallel searches.
    This is necessary because the final writer needs all the results to create the final report.
    """
    _log(f"[collect_results] collecting {len(state.queries_results)} results")
    return {}


# Conditinal Edge
def spawn_researchers(state: ReportState) -> list[Send]:
    return [
        Send("single_search", {"query": query, "user_input": state.user_input})
        for query in state.queries
    ]


# Edges
builder = StateGraph(ReportState)
builder.add_node("build_first_queries", build_first_queries)
builder.add_node("single_search", single_search)
builder.add_node("collect_results", collect_results)
builder.add_node("final_writer", final_writer)

builder.add_edge(START, "build_first_queries")
builder.add_conditional_edges(
    "build_first_queries", spawn_researchers, ["single_search"]
)
builder.add_edge("single_search", "collect_results")
builder.add_edge("collect_results", "final_writer")
builder.add_edge("final_writer", END)

graph = builder.compile()


if __name__ == "__main__":
    st.title("Local AI Research Project")
    user_input = st.text_input(
        "What do you want to research?",
        value="What is the impact of climate change on polar bears?",
    )
    debug_mode = st.checkbox("Show Debug Messages", value=False)

    if st.button("Start Researching"):
        messages = []
        max_messages = int(os.environ.get("MAX_STREAM_MESSAGES", "200"))
        graph_input = {"user_input": user_input, "debug": debug_mode}

        if debug_mode:
            st.markdown("### Debug Stream")
            try:
                count = 0
                for message in graph.stream(graph_input, stream_mode="values"):
                    count += 1
                    messages.append(message)
                    st.write(message)
                    if count >= max_messages:
                        _log(
                            f"[main] reached max_messages ({max_messages}), breaking stream loop"
                        )
                        break
            except Exception as e:
                tb = traceback.format_exc()
                print(tb)
                st.error(f"Error during stream: {e}")
                st.text(tb)
        else:
            with st.spinner("Generating..."):
                try:
                    count = 0
                    for message in graph.stream(graph_input, stream_mode="values"):
                        count += 1
                        messages.append(message)
                        if count >= max_messages:
                            _log(
                                f"[main] reached max_messages ({max_messages}), breaking stream loop"
                            )
                            break
                except Exception as e:
                    tb = traceback.format_exc()
                    print(tb)
                    st.error(f"Error during stream: {e}")
                    st.text(tb)

        def _extract_final(msg):
            if not msg:
                return None
            if isinstance(msg, dict):
                if "final_response" in msg:
                    return msg["final_response"]
                if "value" in msg and isinstance(msg["value"], dict):
                    return msg["value"].get("final_response")
                if "state" in msg and isinstance(msg["state"], dict):
                    return msg["state"].get("final_response")
            return None

        final_response = None

        for m in reversed(messages):
            final_response = _extract_final(m)
            if final_response:
                break

        if final_response:
            st.markdown("### Final Response")
            st.write(final_response)
        else:
            st.warning(
                "No final response found. Check the logs (terminal) and the messages above to diagnose."
            )
