from src.service.graph.router_graph import route_app, RouterState
from src.util.logger import logger

async def start_graph_v2(query: str):
  state = RouterState(
      user_message=query,
  )
  result = await route_app.ainvoke(state)

  return result['bot_message']
