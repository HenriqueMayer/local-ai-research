# Agent logic and graph construction for the local AI research project

from pydantic import BaseModel

from langchain_ollama import ChatOllama
from langgraph.graph import START, END, StateGraph
from langgraph.types import Send

from script.schemas import *
from script.prompts import *

from dotenv import load_dotenv
load_dotenv()

