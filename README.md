Macondo Backend
Macondo Backend is a Python-based service for conversational AI, integrating LangChain, FastAPI, and MCP tools. It supports async agent workflows and tool invocation for structured chat and financial queries.


Features
FastAPI REST endpoints for chat and agent interaction
LangChain agent integration with async support
MCP tool loading and invocation
Docker-ready for local and cloud deployment
Requirements
Python 3.10+
pip install -U langchain-community fastapi uvicorn
MCP service running and accessible
Quick Start
Clone the repo:


git clone https://github.com/mondbond/macondo-be.git
cd macondo-be
Install dependencies:


pip install -r requirements.txt
Start MCP service
Ensure MCP is running and accessible at the configured URL (e.g., http://mond_mcp:8887/mcp).


Run the backend:


uvicorn src.main:app --host 0.0.0.0 --port 8000
Usage
Send POST requests to /chat/ with your message payload.
The backend will process the request using LangChain agents and MCP tools.
