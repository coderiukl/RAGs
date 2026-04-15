import chromadb
from sentence_transformers import SentenceTransformer
from config import EMBED_MODEL, COLLECTION_NAME, TOP_K, MIN_SCORE

_model = None
_collection = None

def _get_model():
    global _model
    if _model is None:
        print("Downloading BGE-M3...")
        _model = SentenceTransformer(EMBED_MODEL)
    return _model

def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient()
        _collection = client.get_or_create_collection(COLLECTION_NAME)
        print(f"Collection: {_collection.count()} chunks")
    return _collection

def retrieve(query: str, top_k: int = TOP_K, keyword: str = None) -> list[dict]:
    model = _get_model()
    collection = _get_collection()

    query_embedding = model.encode(
        [query],
        normalize_embeddings=True,
    ).tolist()

    query_params = {
        "query_embeddings": query_embedding,
        "n_results": top_k,
        "include": ["documents", "metadatas", "distances"],
    }

    if keyword:
        query_params["where_document"] = {"$contains": keyword}
    
    results = collection.query(**query_params)

    chunks = []
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        score = 1 - dist
        
        if score < MIN_SCORE:
            continue

        chunks.append({
            "text": doc,
            "page": meta["page"],
            "source": meta["source"],
            "score": round(score, 4)
        })

    return chunks

def format_context(chunks: list[dict]) -> str:
    if not chunks:
        return "Không tìm thấy thông tin liên quan"
    
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Nguồn_{i} - Trang {chunk['page']} | Score: {chunk['score']}]\n"
            f"{chunk['text']}"
        )
    
    return "\n\n---\n\n".join(parts)

if __name__ == "__main__":
    query = "Quản trị tài chính là gì"
    chunks = retrieve(query)
    print(format_context(chunks))