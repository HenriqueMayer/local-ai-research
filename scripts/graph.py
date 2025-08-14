# Agent logic and graph construction for the local AI research project

from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

from tavily import TavilyClient

import streamlit as st

from schemas import ReportState, QueryResult
from prompts import agent_prompt, build_queries, resume_reasearch, build_final_response, build_final_response

from dotenv import load_dotenv
load_dotenv()


# Models
llm = ChatOllama(model="gemma3:4b")
reasoning_llm = ChatOllama(model="llama3.2:3b")


# Nodes
def build_first_queries(state: ReportState) -> dict: # This dictionary will change the state
    class QueryList(BaseModel):
        queries: list[str]

    query_llm = llm.with_structured_output(QueryList)

    user_input = state.user_input
    prompt = build_queries.format(user_input=user_input)
    result = query_llm.invoke(prompt)

    return {"queries": result.queries}


def single_search(payload: dict) -> dict:
    query = payload["query"]
    user_input = payload["user_input"]
    tavily_client = TavilyClient()

    results = tavily_client.search(
        query, 
        max_results=1,
        include_raw_content=False
    )

    url = results["results"][0]["url"]
    url_extracted = tavily_client.extract(url)

    if len(url_extracted["results"]) > 0:
        raw_content = url_extracted["results"][0]["raw_content"]
        prompt = resume_reasearch.format(user_input=user_input, search_results=raw_content)
        llm_result = llm.invoke(prompt)

    query_result = QueryResult(
        title=results["results"][0]["title"],
        url=url,
        resume=llm_result.content
    )

    return {"queries_results": [query_result]}


def final_writer(state: ReportState):
    search_results = ""
    references = ""
    for i, result in enumerate(state.queries_results):
        search_results += f"[{i+1}]\n\n"
        search_results += f"Title: {result.title}\n"
        search_results += f"URL: {result.url}\n"
        search_results += f"Content: {result.resume}\n"
        search_results += f"================\n\n"

        references += f"[{i+1}] - [{result.title}]({result.url})\n"
    

    prompt = build_final_response.format(user_input=state.user_input,
                                    search_results=search_results)

  
    llm_result = reasoning_llm.invoke(prompt)

    print(llm_result)
    final_response = llm_result.content + "\n\n References:\n" + references
    # print(final_response)

    return {"final_response": final_response}


# Conditinal Edge
def spawn_researchers(state: ReportState) -> list[Send]:
    return [Send("single_search", {"query": query, "user_input": state.user_input})
                 for query in state.queries]


# Edges
builder = StateGraph(ReportState)
builder.add_node("build_first_queries", build_first_queries)
builder.add_node("single_search", single_search)
builder.add_node("final_writer", final_writer)

builder.add_edge(START, "build_first_queries")
builder.add_conditional_edges("build_first_queries", 
                              spawn_researchers, 
                              ["single_search"])
builder.add_edge("single_search", "final_writer")
builder.add_edge("final_writer", END) 

graph = builder.compile()


if __name__ == "__main__":
    st.title("🌎 Local Perplexity")
    user_input = st.text_input("What do you want to research?", 
                               value="What is the impact of climate change on polar bears?")

    if st.button("Start Researching"):
        with st.status("Generating..."):
            messages = []
            for message in graph.stream({"user_input": user_input},
                                      stream_mode="values"):
                messages.append(message)
                st.write(message)
        
        # Get the final message which should contain the final_response
        if messages:
            final_response = messages[-1].get("final_response")
            if final_response:
                st.write(final_response)