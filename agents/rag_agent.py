"""ReAct agent with RAG: policy data is retrieved from two Chroma vector stores.

    member_policy   : member ID -> policy ID(s)
    policy_details  : policy ID -> fees, copays, coverage
"""

import uuid

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver

from agents.react_agent import build_react_graph
from utils.policy_store import member_policy_db, policy_details_db


@tool
def get_member_policies(member_id: str) -> list[str]:
    """Find the policy IDs associated with a member ID (e.g. 'abc123')."""
    docs = member_policy_db.similarity_search(
        f"Member ID {member_id}", k=5, filter={"member_id": member_id.lower()}
    )
    return [d.page_content for d in docs] or [f"No policies found for member {member_id}"]


@tool
def get_policy_details(policy_id: str) -> str:
    """Get fees, co-pays and coverage details for a policy ID (e.g. 'Policy_abc123_1')."""
    docs = policy_details_db.similarity_search(
        f"Policy ID {policy_id}", k=1, filter={"policy_id": policy_id}
    )
    return docs[0].page_content if docs else f"No details found for policy {policy_id}"


tools = [get_member_policies, get_policy_details]

SYSTEM_PROMPT = (
    "You are a helpful insurance customer service assistant. "
    "Use the tools to look up a member's policies and their details before answering. "
    "Base answers only on retrieved policy data; if something isn't covered, say so."
)

checkpointer = InMemorySaver()
rag_graph = build_react_graph(tools, SYSTEM_PROMPT, checkpointer=checkpointer)


if __name__ == "__main__":
    # Each conversation is a thread; the checkpointer keeps its message history
    config = RunnableConfig(configurable={"thread_id": str(uuid.uuid4())})

    questions = [
        "What would be my total payment for a doctor visit? My member id is abc123.",
        "And what about a specialist visit?",  # relies on remembering the member ID
    ]
    for question in questions:
        human_message = HumanMessage(content=question)
        for event in rag_graph.stream({"messages": [human_message]}, config, stream_mode="values"):
            event["messages"][-1].pretty_print()
