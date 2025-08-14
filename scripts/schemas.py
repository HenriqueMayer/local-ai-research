# Define data schemas for agent state and results in the local AI research project

from pydantic import BaseModel
from typing import List
from typing_extensions import Annotated
import operator

class QueryResult(BaseModel):
    title: str = None
    url: str = None
    resume: str = None

class ReportState(BaseModel):
    user_input: str = None
    final_response: str = None
    queries: List[str] = []
    queries_results: Annotated[List[QueryResult], operator.add] # This allows for appending new QueryResult objects to the list