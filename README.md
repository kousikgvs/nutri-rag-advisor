# LLM RAG Advisor 🚀

This project demonstrates a sophisticated Retrieval-Augmented Generation (RAG) pipeline designed to answer questions about LLM-powered autonomous agents. It leverages cutting-edge technologies to provide accurate, context-aware responses by sourcing information from a specific knowledge base.

The application features a command-line interface for direct querying and a user-friendly web interface built with Streamlit, showcasing a complete end-to-end implementation.

## 🌟 Key Features

- **Retrieval-Augmented Generation (RAG):** Enhances Large Language Model (LLM) responses by retrieving relevant information from a specialized vector database.
- **Vector Search:** Utilizes Qdrant and Sentence Transformers for efficient and accurate similarity search.
- **High-Performance Inference:** Powered by the Groq LPU™ Inference Engine for real-time streaming responses.
- **Interactive Frontend:** A simple and intuitive web interface built with Streamlit for easy interaction.
- **Modular & Debug-Friendly:** Code is organized into logical scripts with clear print statements for easy tracking and debugging.

## 🛠️ Tech Stack & Architecture

- **LLM:** Qwen 2 (qwen/qwen3-32b) via Groq
- **Embedding Model:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Database:** Qdrant (Cloud)
- **Application Framework:** Streamlit
- **Core Libraries:** LangChain, PyDotEnv

The architecture is composed of three main parts:
1.  **Indexing:** A script to load, chunk, embed, and index documents into a Qdrant collection.
2.  **Backend Logic:** A script that takes a user query, retrieves relevant context from Qdrant, and generates an answer using the LLM.
3.  **Frontend:** A Streamlit application that provides a UI for the backend logic.

---

## 📂 Project Structure

```
.
├── app.py              # The Streamlit frontend application
├── index_data.py       # Script to index data into the Qdrant vector database
├── query_rag.py        # Script to run the RAG pipeline from the command line
├── .env                # For storing environment variables (API keys)
├── requirements.txt    # Project dependencies
└── README.md           # You are here!
```

### File Descriptions: A Deeper Dive

-   **`index_data.py`**: This script is the foundation of the RAG pipeline. It performs the critical "indexing" stage by connecting to a web source (`WebBaseLoader`) and parsing HTML content with `BeautifulSoup`. The extracted text is then segmented into smaller, overlapping chunks using `RecursiveCharacterTextSplitter`. Each chunk is converted into a high-dimensional vector embedding via the `SentenceTransformer` model. Finally, these vectors, along with their corresponding text payloads, are uploaded to a `Qdrant` vector database, creating a searchable knowledge base.

-   **`query_rag.py`**: This script provides a command-line interface for testing the core RAG logic. When executed, it embeds a hardcoded query, sends it to the Qdrant database to retrieve the most relevant text chunks (the "retrieval" step), and then constructs a detailed prompt that includes this context. This enriched prompt is then sent to the `ChatGroq` LLM, which generates a response that is streamed back to the console. The included print statements make it an invaluable tool for debugging each step of the retrieval and generation process.

-   **`app.py`**: This script brings the project to life with an interactive web interface powered by `Streamlit`. It orchestrates the full user experience, from initializing the models (with `@st.cache_resource` for efficiency) to capturing user input. When a user submits a question, it triggers the same RAG process as `query_rag.py`, but presents the final, streamed response in a clean, user-friendly chat format. It includes spinners and status updates to enhance the user experience during model loading and response generation.

---

## 🚀 Getting Started

Follow these steps to set up and run the project on your local machine.

### 1. Prerequisites

- Python 3.8+
- An account with [Groq](https://console.groq.com/keys) to get an API key.
- An account with [Qdrant Cloud](https://cloud.qdrant.io/) to get a cluster URL and API key.

### 2. Installation & Setup

**1. Clone the repository:**
```bash
git clone <your-repository-url>
cd <your-repository-name>
```

**2. Create and activate a virtual environment:**
```bash
python -m venv venv
source .venv/bin/activate
# On Windows, use: venv\Scripts\activate
```

**3. Install the required dependencies:**
```bash
pip install -r requirements.txt
```

**4. Create a `.env` file** in the root directory of the project and add your API keys and Qdrant URL.

```ini
# .env file
GROQ_API_KEY="your-groq-api-key"
QUADRANT_URL="your-qdrant-cluster-url"
QUADRANT_API_KEY="your-qdrant-api-key"
```

### 3. Execution Steps

**Step 1: Index the Data**

First, you need to populate the Qdrant vector database with the knowledge base. Run the `index_data.py` script. The script includes print statements to track its progress.

```bash
python index_data.py
```

> **⚠️ Important:**
> Run this script only once. Since the collection is not cleared before indexing, running it multiple times will create duplicate entries in your Qdrant vector database.

*This will load data from the specified web path, create embeddings, and store them in your Qdrant collection named `nutrition_rag`.*

**Step 2: (Optional) Test with the Command-Line Interface**

To verify that the RAG pipeline is working correctly, you can run the `query_rag.py` script. This will ask a predefined question and print the answer to your console.

```bash
python query_rag.py
```

**Step 3: Run the Frontend Application**

To start the interactive web interface, run the Streamlit app.

```bash
streamlit run app.py
```

This will launch the application in your web browser. You can now ask any question related to the indexed content!
