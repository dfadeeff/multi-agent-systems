# Multi-Agent Systems — Companion Repository

Companion code repository for the book **_Multi-Agent Systems_** by **Kashaboina**.

This repo contains my working code, notebooks, and experiments while following along with the book — building multi-agent systems with LangChain, LangGraph, MCP, and vector stores.

## Setup

Requires Python 3.12 (some dependencies, e.g. `chromadb`, don't yet support 3.14).

```bash
git clone https://github.com/dfadeeff/multi-agent-systems.git
cd multi-agent-systems

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env   # then add your API keys
```

## Environment variables

API keys live in `.env` (git-ignored). See `.env.example` for the supported variables:
LLM provider/model settings plus OpenAI and Anthropic API keys.

Load them in code with:

```python
from dotenv import load_dotenv
load_dotenv()
```

## Jupyter

Register the venv as a kernel:

```bash
python -m ipykernel install --user --name mas --display-name "Python (mas)"
```

## Stack

- LangChain / LangGraph (+ MCP adapters)
- OpenAI, Anthropic, AWS Bedrock LLM providers
- ChromaDB vector store
- Pydantic, pandas, Jupyter
