# Macondo Backend

Macondo Backend is a Python-based backend service designed for advanced question answering, retrieval-augmented generation (RAG), and evaluation of large language model (LLM) outputs. It provides a modular architecture for integrating various LLM providers, embedding models, and evaluation pipelines, supporting both research and production use cases in natural language processing.

## Features

- **Question Answering & RAG**: Supports context-based answering, financial data extraction, and company news retrieval using LLMs and vector search.
- **Evaluation Suite**: Includes tools for evaluating answer relevance, context relevance, groundedness, and intent classification.
- **Extensible Prompt Management**: Prompts are organized by use case and can be easily extended or customized.
- **Image Embedding**: Supports storing and searching image embeddings for multimodal applications.
- **Chat Memory**: Summarized chat history memory for conversational agents.
- **Agent Routing**: Modular agent routing for different user intentions and tasks.
- **File Format Handling**: Converts and parses various file formats for ingestion and processing.
- **Logging & Configuration**: Centralized logging and environment configuration utilities.

## Project Structure

- `src/`: Main source code, including adapters, database, evaluation, LLM integration, models, services, tools, use cases, and utilities.
- `resources/`: Prompt templates and test data for various tasks and agents.
- `tests/`: Unit and integration tests for core modules.
- `.env` files: Environment variable configuration for different deployment scenarios.

## Dependencies

- **Python 3.10+**
- **FastAPI**: For serving API endpoints.
- **LangChain**: For LLM orchestration and agent management.
- **Pydantic**: For data validation and settings management.
- **Boto3**: For AWS Bedrock and S3 integration.
- **Matplotlib**: For evaluation result visualization.
- **Pillow**: For image processing.
- **Requests**: For HTTP requests.
- **Poetry**: For dependency management.
- **Other**: See `pyproject.toml` and `poetry.lock` for the full list.

---

For more details on specific modules, prompts, or evaluation workflows, refer to the source code and prompt templates in the `resources/` directory.
