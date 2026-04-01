
import os
from dotenv import load_dotenv
import bs4
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# Load environment variables
print("Loading environment variables...")
load_dotenv(override=True)
print("Done loading environment variables.")

groq_api_key = os.getenv("GROQ_API_KEY")
QUADRANT_URL = os.getenv("QUADRANT_URL")
QUADRANT_API_KEY = os.getenv("QUADRANT_API_KEY")

# Only keep post title, headers, and content from the full HTML.
print("Loading documents from web...")
bs4_strainer = bs4.SoupStrainer(class_=("post-title", "post-header", "post-content"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs4_strainer},
)
docs = loader.load()
print("Done loading documents.")

# Split the documents into chunks
print("Splitting documents into chunks...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
chunks = splitter.split_documents(docs)
texts = [doc.page_content for doc in chunks]
print("Done splitting documents.")

# Load the sentence transformer model
print("Loading sentence transformer model...")
model = SentenceTransformer("all-MiniLM-L6-v2")
print("Done loading model.")

# Create vectors
print("Creating vectors...")
vectors = model.encode(texts)
print("Done creating vectors.")

# Connect to Qdrant
print("Connecting to Qdrant...")
qdrant_client = QdrantClient(
    url=QUADRANT_URL,
    api_key=QUADRANT_API_KEY,
)
print("Done connecting to Qdrant.")

# Create collection
#print("Recreating collection...")
#qdrant_client.recreate_collection(
#    collection_name="nutrition_rag",
#    vectors_config=VectorParams(size=384, distance=Distance.COSINE),
#)
#print("Done recreating collection.")

# Create points
print("Creating points...")
points = []
for i, (text, vector) in enumerate(zip(texts, vectors)):
    points.append(
        PointStruct(
            id=i,
            vector=vector.tolist(),  # convert numpy → list
            payload={"text": text}   # store original chunk
        )
    )
print("Done creating points.")

# Upsert points
print("Upserting points to Qdrant...")
qdrant_client.upsert(
    collection_name="nutrition_rag",
    points=points
)
print("Done upserting points.")