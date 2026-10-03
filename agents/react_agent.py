"""ReAct agent: an LLM node that can call tools in a loop until it has an answer.

    START -> invoke_llm --(tool calls?)--> tools -> invoke_llm -> ... -> END
"""

from datetime import date

from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from llm_providers import get_llm

# Simulated external membership system (the book's insurance example)
MEMBERSHIPS = {
    "C001": {"name": "Alice Smith", "status": "Gold", "active": True},
    "C002": {"name": "Bob Jones", "status": "Silver", "active": True},
    "C003": {"name": "Carol White", "status": "Bronze", "active": False},
}


@tool
def get_membership_status(customer_id: str) -> dict:
    """Look up an insurance customer's membership status by customer ID (e.g. 'C001')."""
    return MEMBERSHIPS.get(customer_id.upper(), {"error": f"No customer with ID {customer_id}"})


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


tools = [get_membership_status, multiply, add, get_today]

SYSTEM_PROMPT = (
    "You are a helpful insurance customer service assistant. "
    "Use the available tools to look up customer data or do calculations; "
    "never guess membership details."
)

llm_with_tools = get_llm().bind_tools(tools)


def invoke_llm(state: MessagesState) -> dict:
    messages = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
    return {"messages": [llm_with_tools.invoke(messages)]}


graph_builder = StateGraph(MessagesState)
graph_builder.add_node("invoke_llm", invoke_llm)  # Our LLM node
graph_builder.add_node("tools", ToolNode(tools=tools))  # Prebuilt tool node
graph_builder.add_edge(START, "invoke_llm")
graph_builder.add_conditional_edges("invoke_llm", tools_condition)  # Prebuilt conditional edge
graph_builder.add_edge("tools", "invoke_llm")
react_graph = graph_builder.compile()


if __name__ == "__main__":
    questions = [
        "What is the membership status of customer C001?",
        "Is customer c003 active?",
        "What is (12.5 * 4) + 7?",
        "What's today's date?",
    ]
    for q in questions:
        result = react_graph.invoke({"messages": [("user", q)]})
        print(f"Q: {q}")
        for m in result["messages"][1:-1]:
            print(f"   [{m.type}] {m.tool_calls if m.type == 'ai' else m.content}")
        print(f"A: {result['messages'][-1].content}\n")
