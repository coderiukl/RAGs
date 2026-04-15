import os
from dotenv import load_dotenv

load_dotenv()

COLLECTION_NAME = "rag_basic"
CHROMADB_DIR = "./chroma"
EMBED_MODEL = "BAAI/bge-m3"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = "gemini-2.5-flash"
TOP_K = 10
MIN_SCORE = 0.4
PDF_PATH = "./documents/6quan-tri-tai-chinh.pdf"