import os
import fitz
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import COLLECTION_NAME, EMBED_MODEL, PDF_PATH

model = SentenceTransformer(EMBED_MODEL)

def extract_text_from_pdf(filepath: str) -> list[dict]:
    doc = fitz.open(filepath)
    pages = []
    for i in range(len(doc)):
        text = doc[i].get_text("text").strip()
        if text:
            pages.append({
                "text": text,
                "page": i+1,
                "source": os.path.basename(filepath)
            })
    return pages

def split_into_chunks(pages: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", ",", "!", " ", ""],
    )

    chunks = []
    for page in pages:
        for i, chunk in enumerate(splitter.split_text(page['text'])):
            chunks.append({
                "text": chunk,
                "page": page['page'],
                "source": page['source'],
                "chunk_index": i
            })

    return chunks

def embed_chunks(chunks: list[dict]) -> list[dict]:
    embed_file = model.encode(
        [c['text'] for c in chunks],
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return embed_file.tolist()

def save_to_chromadb(chunks, embeddings):
    client = chromadb.PersistentClient()
    try:
        client.delete_collection(COLLECTION_NAME)
    except:
        pass

    collection = client.create_collection(
        COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    for i in range(0, len(chunks), 500):
        bc = chunks[i : i + 500]
        be = embeddings[i : i + 500]

        collection.add(
            ids=[f"Nguồn_{i+j}" for j in range(len(bc))],
            embeddings = be,
            documents = [c['text'] for c in bc],
            metadatas = [{"page": c['page'], "source": c['source']} for c in bc],
        )
    print(f"Saved {collection.count()} chunks into vector DB")

def index_multiple_pdfs(filepaths: list[str]):
    all_chunks = []

    for pdf_path in filepaths:
        print(f"Processing: {os.path.basename(pdf_path)}")
        pages = extract_text_from_pdf(pdf_path)
        chunks = split_into_chunks(pages)
        all_chunks.extend(chunks)
        print(f"{len(chunks)} chunks")
    
    print(f"\nTotal: {len(all_chunks)} chunks from {len(filepaths)} file")

    embeddings = embed_chunks(all_chunks)
    save_to_chromadb(all_chunks, embeddings)

    
def main():
    pages = extract_text_from_pdf(PDF_PATH)
    chunks = split_into_chunks(pages)
    embeddings = embed_chunks(chunks)
    save_to_chromadb(chunks, embeddings)
    print("Index Process Completed")


if __name__ == "__main__":
    main()