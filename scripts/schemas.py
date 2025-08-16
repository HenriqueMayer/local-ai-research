# Define data schemas for agent state and results in the local AI research project

from pydantic import BaseModel, Field
from typing import List, Optional, Annotated
import operator


class QueryResult(BaseModel):
    title: Optional[str] = None
    url: Optional[str] = None
    resume: Optional[str] = None


class ReportState(BaseModel):
    user_input: Optional[str] = None
    final_response: Optional[str] = None
    queries: List[str] = Field(default_factory=list)
    queries_results: Annotated[List[QueryResult], operator.add] = Field(
        default_factory=list
    )
    debug: bool = False
