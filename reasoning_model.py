# reasoning_model.py
from smolagents import OpenAIServerModel, CodeAgent
import os

# Define local model names
reasoning_model_id = "gemma2:2b"  # Use DeepSeek for reasoning

def get_model(model_id):
    """Returns an Ollama model."""
    return OpenAIServerModel(
        model_id=model_id,
        api_base="http://localhost:11434/v1",  # Ollama API endpoint
        api_key="ollama"
    )

# Create reasoning model using DeepSeek
reasoning_model = get_model(reasoning_model_id)

# Create reasoner agent
reasoner = CodeAgent(tools=[], model=reasoning_model, add_base_tools=False, max_steps=2)