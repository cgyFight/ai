import chunks
import chromadb

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("DASHSCOPE_API_KEY")

openai_client = OpenAI(
    api_key=api_key,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

def embed_text(text: str) -> list[float]:
    """Generates embeddings for the given text using OpenAI embeddings API."""
    response = openai_client.embeddings.create(
        model="text-embedding-v4",  # Updated to a supported model
        input=text
    )
    return response.data[0].embedding

def create_chroma_collection(collection_name: str):
    """Creates a ChromaDB collection."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name=collection_name)
    return collection

def create_and_store_embeddings(file_path: str, collection_name: str):
    """Reads data from a file, splits it into chunks, generates embeddings, and stores them in ChromaDB."""
    data = chunks.read_data(file_path)
    chunk_list = chunks.combine_chunks_when_start_with(
        chunks.split_into_chunks(data),
        start_str="#"
    )

    collection = create_chroma_collection(collection_name)

    for i, chunk in enumerate(chunk_list):
        embedding = embed_text(chunk)
        collection.upsert(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"chunk_{i}"]
        )
    print(f"Stored {len(chunk_list)} chunks in collection '{collection_name}'.")

def query_similar_chunks(query: str, collection_name: str, n_results: int = 3) -> list[str]:
    """Queries the ChromaDB collection for similar chunks based on the input query."""
    collection = create_chroma_collection(collection_name)
    query_embedding = embed_text(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results["documents"][0]


if __name__ == "__main__":
    file_path = "data.md"
    collection_name = "demo_collection"

    # Create and store embeddings
    # create_and_store_embeddings(file_path, collection_name)

    # Query similar chunks
    question = "令狐冲领悟了什么魔法？"
    similar_chunks = query_similar_chunks(question, collection_name)
    print("Similar Chunks:")
    for chunk in similar_chunks:
        print("---- Chunk ----")
        print(chunk)
        print("----------------\n")

    prompt = f"根据以下内容回答问题：\n\n{''.join(similar_chunks)}\n\n问题：{question}"

    response = openai_client.chat.completions.create(
        model="qwen-plus",
        messages=[
            {"role": "system", "content": "You are an experienced programmer."},
            {"role": "user", "content": prompt}
        ]
    )
    print("Response:")
    print(response.choices[0].message.content)

    




