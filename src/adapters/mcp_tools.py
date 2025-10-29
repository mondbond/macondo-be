
def get_mcp_config():
   mcp_config = {
       "servers": [
           {
               "name": "mond_mcp",
               "transport": "streamable-http",
               "url": "http://mond_mcp:8887",
               "description": (
                   "This server provides tools to get stock price changes and fundamental properties. "
                   "Available tools:\n"
                   "1. get_ticker_price_change(ticker: str) -> dict: Get price change for a ticker in percents for the last trading day.\n"
                   "2. get_ticker_fundamental(ticker: str, fund_property: str) -> dict: Get fundamental property for a ticker. "
                   "Available properties include marketCap, trailingPE, forwardPE, priceToBook, beta, dividendYield, earningsPerShare."
               )
           }
       ]
   }

   return mcp_config
