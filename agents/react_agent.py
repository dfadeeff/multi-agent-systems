"""ReAct agent: an LLM node that can call tools in a loop until it has an answer.

    START -> invoke_llm --(tool calls?)--> tools -> invoke_llm -> ... -> END
"""

from datetime import date

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from llm_providers import get_llm

# Simulated external membership system (the book's insurance example)
MEMBERSHIPS = {
    "C001": {"name": "Alice Smith", "status": "Gold", "active": True},
    "C002": {"name": "Bob Jones", "status": "Silver", "active": True},
    "C003": {"name": "Carol White", "status": "Bronze", "active": False},
    "ABC123": {"name": "Dan Brown", "status": "Silver", "active": True},
}

# Copay per visit type, by membership tier
COPAYS = {
    "Gold": {"doctor_visit": 10, "specialist_visit": 25, "emergency_room": 100},
    "Silver": {"doctor_visit": 25, "specialist_visit": 50, "emergency_room": 200},
    "Bronze": {"doctor_visit": 40, "specialist_visit": 80, "emergency_room": 350},
}


@tool
def get_membership_status(customer_id: str) -> dict:
    """Look up an insurance customer's membership status by customer/member ID (e.g. 'C001')."""
    return MEMBERSHIPS.get(customer_id.upper(), {"error": f"No customer with ID {customer_id}"})


@tool
def get_copay(membership_status: str, visit_type: str) -> dict:
    """Get the copay in USD for a visit type ('doctor_visit', 'specialist_visit',
    'emergency_room') under a membership tier ('Gold', 'Silver', 'Bronze')."""
    tier = COPAYS.get(membership_status.capitalize())
    if tier is None or visit_type not in tier:
        return {"error": f"Unknown tier {membership_status!r} or visit type {visit_type!r}"}
    return {"membership_status": membership_status, "visit_type": visit_type, "copay_usd": tier[visit_type]}


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@tool
def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b


@tool
def get_today() -> str:
    """Return today's date in ISO format."""
    return date.today().isoformat()


tools = [get_membership_status, get_copay, multiply, add, get_today]

SYSTEM_PROMPT = (
    "You are a helpful insurance customer service assistant. "
    "Use the available tools to look up customer data or do calculations; "
    "never guess membership details."
)


def build_react_graph(tools: list, system_prompt: str, checkpointer=None):
    """Compile a ReAct graph: LLM node <-> prebuilt tool node, looping until no tool calls.

    Pass a checkpointer (e.g. InMemorySaver) to persist conversation state per thread_id.
    """
    llm_with_tools = get_llm().bind_tools(tools)

    def invoke_llm(state: MessagesState) -> dict:
        messages = [SystemMessage(content=system_prompt), *state["messages"]]
        return {"messages": [llm_with_tools.invoke(messages)]}

    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("invoke_llm", invoke_llm)  # Our LLM node
    graph_builder.add_node("tools", ToolNode(tools=tools))  # Prebuilt tool node
    graph_builder.add_edge(START, "invoke_llm")
    graph_builder.add_conditional_edges("invoke_llm", tools_condition)  # Prebuilt conditional edge
    graph_builder.add_edge("tools", "invoke_llm")
    return graph_builder.compile(checkpointer=checkpointer)


react_graph = build_react_graph(tools, SYSTEM_PROMPT)


if __name__ == "__main__":
    question = "What would be my total payment for a doctor visit? My member id is abc123."
    human_message = HumanMessage(content=question)
    # stream_mode="values" yields the full graph state after each step
    for event in react_graph.stream({"messages": [human_message]}, stream_mode="values"):
        event["messages"][-1].pretty_print()
