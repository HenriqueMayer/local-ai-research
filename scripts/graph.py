# Agent logic and graph construction for the local AI research project

from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

from tavily import TavilyClient

from schemas import ReportState, QueryResult
from prompts import agent_prompt, build_queries, resume_reasearch

from dotenv import load_dotenv
load_dotenv()


# Models
llm = ChatOllama(model="gemma3:4b")
reasoning_llm = ChatOllama(model="llama3.2:3b")


# Node
def build_first_queries(state: ReportState) -> dict: # This dictionary will change the state
    class QueryList(BaseModel):
        queries: List[str]

    query_llm = llm.with_structured_output(QueryList)

    user_input = state.user_input
    prompt = build_queries.format(user_input=user_input)
    result = query_llm.invoke(prompt)

    return {"queries": [result.queries]}


def single_search(query: str) -> dict:
    tavily_client = TavilyClient()

    results = tavily_client.search(
        query, 
        max_results=1,
        include_raw_content=False
    )

    url = results["results"][0]["url"]
    url_extracted = travily_client.extract(url)

    if len(url_extracted["results"]) > 0:
        raw_content = url_extracted["results"][0]["raw_content"]
        prompt = resume_reasearch.format(user_input=user_input, search_results=raw_content)
        llm_result = llm.invoke(prompt)

    query_result = QueryResult(
        title=results["results"][0]["title"],
        url=url,
        resume=llm_result.content
    )

    return {"query_result": [query_result]}

# Conditinal Edge
def spawn_reasearchers(state: ReportState) -> list[Send]:
    return [Send("single_serach", query)
                 for query in state.queries]


# Edges
builder = StateGraph(ReportState)
graph = builder.compile()


if __name__ == "__main__":

    user_input = """
    Can you exaplain to me how is the full process of building an LLM from scratch?
    """

    graph.invoke({"user_input": user_input})