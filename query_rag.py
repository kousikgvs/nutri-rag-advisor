
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

# Load environment variables
print("Loading environment variables...")
load_dotenv(override=True)
print("Done loading environment variables.")

groq_api_key = os.getenv("GROQ_API_KEY")
QUADRANT_URL = os.getenv("QUADRANT_URL")
QUADRANT_API_KEY = os.getenv("QUADRANT_API_KEY")


# Initialize the LLM
print("Initializing LLM...")
llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=groq_api_key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)
print("Done initializing LLM.")

# Load the sentence transformer model
print("Loading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Done loading model.")

# Connect to Qdrant
print("Connecting to Qdrant...")
qdrant_client = QdrantClient(
    url=QUADRANT_URL,
    api_key=QUADRANT_API_KEY,
)
print("Done connecting to Qdrant.")

# Query
query = "what are the key components of an LLM-powered agent?"
print(f"Query: {query}")

print("Encoding query...")
query_vector = model.encode(query).tolist()
print("Done encoding query.")

print("Searching for similar documents in Qdrant...")
results = qdrant_client.search(
    collection_name="nutrition_rag",
    query_vector=query_vector,
    limit=3
)
print("Done searching for documents.")

print("Constructing context from search results...")
context = ""
for res in results:
    context += res.payload["text"] + "\n\n"
print("Done constructing context.")

print("Creating prompt for the LLM...")
system_message = "You are a helpful assistant. Based on the following context, please answer the user's question. \n\nContext:\n" + context

messages = [
    SystemMessage(content=system_message),
    HumanMessage(content=query)
]
print("Done creating prompt.")

# Stream the response
print("Streaming response from LLM...")
for chunk in llm.stream(messages):
    print(chunk.content, end="")
print("\nDone streaming response.")
