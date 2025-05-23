import os
from datetime import datetime
from typing import Dict

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import MessagesState, StateGraph, START
from langgraph.prebuilt import ToolNode, tools_condition
from dotenv import load_dotenv

load_dotenv()


@tool
def get_current_time() -> Dict[str, str]:
    """Return the current UTC time in ISO-8601 format.
    Example → {"utc": "2025-05-21T06:42:00Z"}
    """
    current_time = datetime.now()
    return {"utc": current_time.strftime("%Y-%m-%dT%H:%M:%SZ")}


tools = [get_current_time]

llm = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL_NAME"),
    temperature=0.1,
    api_key=os.getenv("OPENAI_API_KEY")
).bind_tools(tools)


def chatbot_node(state: MessagesState) -> MessagesState:
    """Main chatbot logic node"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}


tool_node = ToolNode(tools)

graph_builder = StateGraph(MessagesState)

graph_builder.add_node("chatbot", chatbot_node)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "chatbot")

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)

graph_builder.add_edge("tools", "chatbot")

app = graph_builder.compile()

if __name__ == "__main__":
    print("Chat bot started! Ask me anything or ask for the time.")
    print("Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break

        if not user_input:
            continue

        initial_state = {"messages": [HumanMessage(content=user_input)]}

        try:
            result = app.invoke(initial_state)

            for message in reversed(result["messages"]):
                if isinstance(message, AIMessage):
                    print(f"Bot: {message.content}")
                    break

        except Exception as e:
            print(f"Error: {e}")

        print()