# primary_agent.py
from smolagents import OpenAIServerModel, ToolCallingAgent
from smolagents.tools import Tool
from rag_tool import rag_with_reasoner
import os

class RAGReasoningTool(Tool):
    name = "rag_reasoning_tool"
    description = """
    Searches a vector database for relevant context and uses a reasoning model
    to generate an informative, concise response to the user's question.
    """
    inputs = {
        "user_query": {
            "type": "string",
            "description": "The user's question to be answered based on retrieved context."
                           ,
        }
    }
    output_type = "string"

    def forward(self, user_query: str):
        return rag_with_reasoner(user_query)

# Instantiate the tool
rag_tool = RAGReasoningTool()

def get_model(model_id):
    """Returns an Ollama model."""
    return OpenAIServerModel(
        model_id=model_id,
        api_base="http://localhost:11434/v1",  # Ollama API endpoint
        api_key="ollama"
    )

# Load tool model (Qwen-2.5:7b)
primary_model = get_model("qwen2.5:3b")

# Create primary agent using Qwen for tool responses
primary_agent = ToolCallingAgent(tools=[rag_tool], model=primary_model, add_base_tools=False, max_steps=3)

# Export the agent to be used in the app
def get_primary_agent():
    return primary_agent