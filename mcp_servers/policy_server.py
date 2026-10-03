"""MCP server exposing the member/policy retrieval tools over streamable HTTP.

Run:  python -m mcp_servers.policy_server   ->  http://127.0.0.1:8000/mcp
"""

from mcp.server.fastmcp import FastMCP

from utils.policy_store import member_policy_db, policy_details_db

# Initialize MCP server (book uses host="0.0.0.0"; localhost keeps it off the network)
mcp = FastMCP(name="member-policy-mcp", host="127.0.0.1", port=8000)


@mcp.tool()
def get_member_policy_id(member_id: str) -> str:
    """Find the policy ID(s) associated with a member ID (e.g. 'abc123')."""
    docs = member_policy_db.similarity_search(
        f"Member ID {member_id}", k=5, filter={"member_id": member_id.lower()}
    )
    if not docs:
        return f"No policies found for member {member_id}"
    return ", ".join(d.metadata["policy_id"] for d in docs)


@mcp.tool()
def get_policy_details(policy_id: str) -> str:
    """Get fees, co-pays and coverage details for a policy ID (e.g. 'Policy_abc123_1')."""
    docs = policy_details_db.similarity_search(
        f"Policy ID {policy_id}", k=1, filter={"policy_id": policy_id}
    )
    return docs[0].page_content if docs else f"No details found for policy {policy_id}"


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
