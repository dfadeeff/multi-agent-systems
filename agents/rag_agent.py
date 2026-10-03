"""ReAct agent with RAG: policy data is retrieved from two Chroma vector stores.

    member_policy   : member ID -> policy ID(s)
    policy_details  : policy ID -> fees, copays, coverage
"""

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

from agents.react_agent import build_react_graph

# Embeddings for Chroma
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Sample member -> policy mapping
sample_member_policy_docs = [
    Document(
        page_content="Member ID abc123 is associated with policy Policy_abc123_1.",
        metadata={"member_id": "abc123", "policy_id": "Policy_abc123_1"},
    ),
    Document(
        page_content="Member ID xyz789 is associated with policy Policy_xyz789_1.",
        metadata={"member_id": "xyz789", "policy_id": "Policy_xyz789_1"},
    ),
    Document(
        page_content="Member ID xyz789 is associated with policy Policy_xyz789_2.",
        metadata={"member_id": "xyz789", "policy_id": "Policy_xyz789_2"},
    ),
]

# Sample policy -> policy details
sample_policy_detail_docs = [
    Document(
        page_content=(
            "Policy ID Policy_abc123_1. "
            "Doctor visit fee is $100 and Co-Pay is $5. "
            "Specialist visit fee is $250 and Co-Pay is $30. "
            "Annual deductible is $500 and has been met."
        ),
        metadata={"policy_id": "Policy_abc123_1", "plan": "Gold"},
    ),
    Document(
        page_content=(
            "Policy ID Policy_xyz789_1. "
            "Doctor visit fee is $100 and Co-Pay is $25. "
            "Specialist visit fee is $250 and Co-Pay is $60. "
            "Annual deductible is $1500 and has not been met."
        ),
        metadata={"policy_id": "Policy_xyz789_1", "plan": "Silver"},
    ),
    Document(
        page_content=(
            "Policy ID Policy_xyz789_2. "
            "Dental policy. Cleaning fee is $120 and Co-Pay is $10. "
            "Filling fee is $200 and Co-Pay is $40."
        ),
        metadata={"policy_id": "Policy_xyz789_2", "plan": "Dental"},
    ),
]

member_policy_db = Chroma.from_documents(
    documents=sample_member_policy_docs,
    embedding=embeddings,
    collection_name="member_policy",
)
policy_details_db = Chroma.from_documents(
    documents=sample_policy_detail_docs,
    embedding=embeddings,
    collection_name="policy_details",
)


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

rag_graph = build_react_graph(tools, SYSTEM_PROMPT)


if __name__ == "__main__":
    question = "What would be my total payment for a doctor visit? My member id is abc123."
    human_message = HumanMessage(content=question)
    for event in rag_graph.stream({"messages": [human_message]}, stream_mode="values"):
        event["messages"][-1].pretty_print()
