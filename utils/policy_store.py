"""Sample insurance data in two Chroma vector stores (in-memory).

    member_policy_db   : member ID -> policy ID(s)
    policy_details_db  : policy ID -> fees, copays, coverage
"""

from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

load_dotenv()  # OPENAI_API_KEY for embeddings (also needed when run as the MCP server)

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
