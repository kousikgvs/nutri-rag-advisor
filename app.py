
import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Load environment variables
load_dotenv(override=True)

groq_api_key = os.getenv("GROQ_API_KEY")
QUADRANT_URL = os.getenv("QUADRANT_URL")
QUADRANT_API_KEY = os.getenv("QUADRANT_API_KEY")

# --- App Setup ---
st.title("LLM RAG Advisor")
st.write("Ask me anything about LLM-powered autonomous agents!")

# --- Model and Client Initialization ---

@st.cache_resource
def get_llm():
    return ChatGroq(
        model="qwen/qwen3-32b",
        api_key=groq_api_key,
        temperature=0,
    )

@st.cache_resource
def get_retriever_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=QUADRANT_URL,
        api_key=QUADRANT_API_KEY,
    )

# Initialize each component with its own spinner
with st.spinner("Initializing LLM..."):
    llm = get_llm()

with st.spinner("Initializing Retriever Model... (This can take a while on the first run)"):
    model = get_retriever_model()

with st.spinner("Connecting to Vector DB..."):
    qdrant_client = get_qdrant_client()

st.success("Ready to answer your questions!")

# --- RAG Logic ---
def get_rag_response(query):
    query_vector = model.encode(query).tolist()

    results = qdrant_client.search(
        collection_name="nutrition_rag",
        query_vector=query_vector,
        limit=3
    )

    context = ""
    for res in results:
        context += res.payload["text"] + "\n\n"

    system_message = """You are a helpful assistant. Based on the following context, 
    please answer the user's question. If the context does not provide an answer, 
    say so.

    Context:
    """ + context

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=query)
    ]

    # Stream the response from the LLM
    for chunk in llm.stream(messages):
        yield chunk.content

# --- Streamlit UI ---
user_query = st.text_input("Enter your question:", "")

if st.button("Ask"):
    if user_query:
        with st.spinner("Thinking... Give me a moment.."):
            response_generator = get_rag_response(user_query)
            st.write_stream(response_generator)
    else:
        st.warning("Please enter a question.")
