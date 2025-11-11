from enum import Enum
from pydantic import BaseModel, Field
from src.util.logger import logger

class UserIntentionEnum(str, Enum):
  NEWS_ABOUT_COMPANY = "NEWS_ABOUT_COMPANY"
  COMPANY_INFORMATION_FROM_REPORT = "COMPANY_INFORMATION_FROM_REPORT"
  OTHER_FINANCIAL_QUESTIONS = "OTHER_FINANCIAL_QUESTIONS"
  ANALYSE_SHARE_PRISE = "ANALYSE_SHARE_PRISE"

  @classmethod
  def from_str(cls, value: str):
    for member in cls:
      if member.value.lower() == value.lower():
        return member
    raise ValueError(f"Unknown intention: {value}")

class RouterDto(BaseModel):
  """
RouterDto is the structured output that determines:
- what company ticker the user is referring to (if any)
- what the user's high-level intention is regarding that ticker
"""

  ticker: list[str] = Field(
        description=(
          "Stock ticker symbols (short code like AAPL, GOOGL, MSFT) explicitly "
          "mentioned by the user in their query."
        ),
        examples=["AAPL", "GOOGL", "MSFT"],
        max_length=5,
        min_length=0
    )

  intention: UserIntentionEnum = Field(
      description=(
        "Classify the user's high-level intention:\n"
        "- OTHER_FINANCIAL_QUESTIONS -> Default option if any of the above is not strictly related. Choose this option even if question is not related to financials\n"
        "- COMPANY_INFORMATION_FROM_REPORT → User wants **specific financial/report data** from an company report (requires ticker). Choose this when user ask you to get something from annual report strictly\n"
        "- ANALYSE_SHARE_PRISE → User wants to understand the reason for last share prices drop. Choose this strictly when user says to *analyze* the recent stock price change\n"
        "- NEWS_ABOUT_COMPANY → User wants recent news or updates about a specific company (requires ticker) from the news."
      )
  )

  image_wanted: bool = Field(
      description=(
        "Indicates whether the user desires visual content (like images charts or graphs) in the response. True if user explicitly requests visual data, False otherwise."
      )
  )

