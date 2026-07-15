# ROS (Release Build)
An advanced, multi-tenant AI-assisted Legal Research and Metadata Extraction Portal.

ROS brings together SQLite-powered keyword indexers, semantic vector retrievers (BGE-M3/Chroma DB), and state-of-the-art LLMs (Anthropic Claude API & Local GGUF) into a single, cohesive legal workspace.

## Key Features
*   **AI Legal Chat (Ask ROS)**: Real-time legal research assistant powered by Claude or local models, leveraging case law context retrieval.
*   **Case Breakdown Analyzer**: Streamlined Pydantic-validated ratio decidendi analysis, legal extraction, and academic synthesis.
*   **Batch Ingest & Extractor Portal**: High-speed, multi-threaded metadata extraction queue with single-file and batch "Add to Corpus" ingestion.
*   **Multi-tenant User & Role Management**:
    *   **Administrator**: Full access to global LLM config, database sync, and user account creation.
    *   **Research Officer (RO)**: Permissions of a Legal Researcher plus active extraction portal upload access.
    *   **Legal Researcher**: Access to semantic search, bookmarks, and Ask ROS conversational tools.
    *   **Guest**: Read-only access to case judgments and search.
*   **Aesthetic UI**: Curated crimson sidebar, glassmorphic header search, responsive dual-pane layout, and rich visual signifiers.

## Installation & Setup
1. Clone the repository and navigate to the folder:
   ```bash
   git clone https://github.com/laksh213/ROS-Release-Build.git
   cd ROS-Release-Build
   ```
2. Create and activate a python virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy the environment template and insert your API keys:
   ```bash
   cp .env.example .env
   ```
5. Launch the application:
   ```bash
   ./roscribe_smart.sh start --public
   ```

## Architecture
*   **Backend**: Python, FastAPI, NiceGUI
*   **Search Core**: SQLite FTS5 (Sparse Indexing) + Chroma DB (Dense Indexing)
*   **Model Integration**: Anthropic SDK (Claude 3.5 Sonnet / Haiku) + Llama.cpp (Local GGUF execution)
