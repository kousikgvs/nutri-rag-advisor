
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


# Initialize the LLM
llm = ChatGroq(
    model="qwen/qwen3-32b",
    api_key=groq_api_key,
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# Load the sentence transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Connect to Qdrant
qdrant_client = QdrantClient(
    url=QUADRANT_URL,
    api_key=QUADRANT_API_KEY,
)

# Query
query = "foods for anemia"

query_vector = model.encode(query).tolist()

results = qdrant_client.search(
    collection_name="nutrition_rag",
    query_vector=query_vector,
    limit=3
)

for res in results:
    print(res.payload["text"])
