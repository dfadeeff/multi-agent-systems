"""ReAct agent whose tools live on a remote MCP server (mcp_servers/policy_server.py).

Start the server first:  python -m mcp_servers.policy_server
Then run the agent:      python -m agents.mcp_agent
"""

import asyncio
import uuid

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.memory import InMemorySaver

from agents.react_agent import build_react_graph

MCP_SERVERS = {
    "member-policy": {
        "url": "http://127.0.0.1:8000/mcp",
        "transport": "streamable_http",
    },
}

SYSTEM_PROMPT = (
    "You are a helpful insurance customer service assistant. "
    "Use the tools to look up a member's policies and their details before answering. "
    "Base answers only on retrieved policy data; if something isn't covered, say so."
)


async def build_mcp_agent(checkpointer=None):
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()  # MCP tools wrapped as LangChain tools
    return build_react_graph(tools, SYSTEM_PROMPT, checkpointer=checkpointer)


async def main():
    agent = await build_mcp_agent(checkpointer=InMemorySaver())
    config = RunnableConfig(configurable={"thread_id": str(uuid.uuid4())})

    question = "What would be my total payment for a doctor visit? My member id is abc123."
    human_message = HumanMessage(content=question)
    async for event in agent.astream({"messages": [human_message]}, config, stream_mode="values"):
        event["messages"][-1].pretty_print()


if __name__ == "__main__":
    asyncio.run(main())
