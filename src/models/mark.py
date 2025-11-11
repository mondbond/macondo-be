from pydantic import BaseModel, Field


class ZeroToTenMark(BaseModel):
  """Schema for marking answers from 0 to 10."""
  mark: int = Field(
    description="Provide a score from 0 to 10 based on its quality of answering the question, where 10 is the best possible answer and 0 is the worst possible answer.")
