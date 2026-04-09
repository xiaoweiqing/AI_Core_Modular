# importer.py (v2 - with Proxy Cleaner)
# This is the ONLY script you need to index your constitution.
import os
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import MarkdownTextSplitter # We use this for better chunking
from pathlib import Path
import uuid

# --- [CRITICAL] Proxy Cleaner ---
# Ensures system proxies don't interfere with the local connection
for proxy_var in ["http_proxy", "https_proxy", "all_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"]:
    if proxy_var in os.environ:
        print(f">> [Proxy Cleaner] Found and removed system proxy: {proxy_var}")
        del os.environ[proxy_var]
# --- Proxy Cleaner End ---

# --- Configuration ---
QDRANT_CLIENT = QdrantClient("localhost", port=6333)
EMBEDDING_MODEL = SentenceTransformer(str(Path(__file__).parent / "all-MiniLM-L6-v2"))
COLLECTION_NAME = "personal_constitution"
# The path to your single, authoritative constitution file
CONSTITUTION_FILE_PATH = Path(__file__).parent / "personal_constitution" / "my_core_principles.md"


def import_constitution(file_path: Path):
    try:
        print(f"Reading new constitution file: {file_path}")
        if not file_path.exists():
            print(f"❌ ERROR: Constitution file not found at {file_path}")
            return

        text = file_path.read_text(encoding='utf-8')

        # We split the constitution by articles to make the retrieval more precise
        splitter = MarkdownTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_text(text)
        print(f"Constitution has been split into {len(chunks)} principle chunks.")

        # Use recreate_collection to ensure we always start with a clean slate
        QDRANT_CLIENT.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE),
        )
        print(f"Created a fresh, new collection: '{COLLECTION_NAME}'")

        print("Vectorizing and uploading principle chunks...")
        points_to_upsert = []
        for i, chunk in enumerate(chunks):
            vector = EMBEDDING_MODEL.encode(chunk).tolist()
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{file_path.name}-{i}"))
            points_to_upsert.append(
                models.PointStruct(id=point_id, vector=vector, payload={"text_chunk": chunk})
            )

        if points_to_upsert:
            QDRANT_CLIENT.upsert(
                collection_name=COLLECTION_NAME,
                points=points_to_upsert,
                wait=True
            )
            print(f"✅✅ Success! {len(points_to_upsert)} principle chunks have been injected into your AI's brain.")

    except Exception as e:
        print(f"❌ Import Failed: {e}")
        print("   Please ensure the Qdrant Docker container is running.")

if __name__ == "__main__":
    import_constitution(CONSTITUTION_FILE_PATH)
