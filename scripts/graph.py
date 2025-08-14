# Agent logic and graph construction for the local AI research project

from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

from scripts.schemas import ReportState, QueryResult
from scripts.prompts import *

from dotenv import load_dotenv
load_dotenv()

# Models
llm = ChatOllama(model="gemma3:4b")
reasoning_llm = ChatOllama(model="llama3.2:3b")


# Node (to do)


# Edges
builder = StateGraph(ReportState)
graph = builder.compile()





if __name__ == "__main__":

    user_input = """
    Can you exaplain to me how is the full process of building an LLM from scratch?
    """

    graph.invoke({"user_input": user_input})