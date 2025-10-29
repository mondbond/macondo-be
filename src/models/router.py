from enum import Enum
from pydantic import BaseModel, Field

class UserIntentionEnum(str, Enum):
  # NEWS_ABOUT_COMPANY = "NEWS_ABOUT_COMPANY"
  COMPANY_INFORMATION_FROM_REPORT = "COMPANY_INFORMATION_FROM_REPORT"
  NOT_RELATED = "NOT_RELATED"
  # ANALYSE_SHARE_PRISE = "ANALYSE_SHARE_PRISE"

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
        min_length=1
    )

  intention: UserIntentionEnum = Field(
      description=(
        "Classify the user's high-level intention:\n"
        "- NOT_RELATED -> Default option if the query is unrelated to stocks, tickers, or finance (e.g. 'What is the capital of France?'). Not choose it for finance-related queries.\n"
        "- COMPANY_INFORMATION_FROM_REPORT → User wants **specific financial/report data** from an company report (requires ticker).\n"
        # "- ANALYSE_SHARE_PRISE → User wants to understande the reason for last share prices.\n"
      )
  )
