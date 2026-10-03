# Policy MCP server image.
# Build (from repo root):  docker build -f mcp_servers/Dockerfile -t policy-mcp .
# Run:                     docker run --rm -p 8000:8000 --env-file .env policy-mcp
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8000

WORKDIR /app

COPY mcp_servers/requirements.txt mcp_servers/requirements.txt
RUN pip install --no-cache-dir -r mcp_servers/requirements.txt

# Only the code the server needs
COPY utils/ utils/
COPY mcp_servers/ mcp_servers/

# Run as non-root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["python", "-m", "mcp_servers.policy_server"]
