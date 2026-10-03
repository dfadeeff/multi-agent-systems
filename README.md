# Practical Multi-Agent AI Systems — Companion Repository

Companion code repository for the book **_Practical Multi-Agent AI Systems: How to Architect, Build, and Scale Next-Generation AI Systems That Work in the Real World_** (Tech Today) by **Kashaboina**.

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

## Project structure

```
agents/          # agent implementations
llm_providers/   # get_llm() factory for OpenAI / Anthropic chat models
mcp_servers/     # MCP servers exposing tools over streamable HTTP
utils/           # shared helpers
config/          # settings loaded from .env
notebooks/       # Jupyter notebooks per chapter
data/            # local datasets and vector stores
tests/           # tests
```

Quick check:

```python
from llm_providers import get_llm

llm = get_llm()  # uses LLM_PROVIDER / LLM_MODEL from .env
print(llm.invoke("Hello!").content)
```

## MCP server

Run locally:

```bash
python -m mcp_servers.policy_server   # http://127.0.0.1:8000/mcp
python -m agents.mcp_agent            # in another terminal
```

Or in Docker (the API key is passed at runtime, never baked into the image):

```bash
docker build -f mcp_servers/Dockerfile -t policy-mcp .
docker run --rm -p 8000:8000 --env-file .env policy-mcp
```

## Stack

- LangChain / LangGraph (+ MCP adapters)
- OpenAI, Anthropic, AWS Bedrock LLM providers
- ChromaDB vector store
- Pydantic, pandas, Jupyter
