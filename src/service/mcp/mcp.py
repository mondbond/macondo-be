from langchain_mcp_adapters.client import MultiServerMCPClient

from src.util.env_property import get_env_property
from src.util.logger import logger

mcp_url = get_env_property("MCP_FIN_URL")
logger.info(f"MCP_FIN_URL: {mcp_url}")


mcp_client = None

# http://mond_mcp:8887/mcp
if mcp_url is not None and mcp_url != "":
  logger.info("MCP_FIN_URL:", mcp_url)
  mcp_client = MultiServerMCPClient(
      {

        "mond_mcp": {
          "transport": "streamable_http",
          "url": mcp_url,
        }
      }
  )
else:
  logger.warning("No MCP_FIN_URL provided, skipping MCP tools loading.")
