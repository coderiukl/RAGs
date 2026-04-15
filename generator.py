import google.generativeai as genai
from retriever import retrieve, format_context
from config import GEMINI_API_KEY, GEMINI_MODEL, TOP_K

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel(GEMINI_MODEL)

def build_prompt(query: str, chunks: list[dict]) -> str:
    context = format_context(chunks)
    return f"""Bạn là trợ lý AI chuyên trả lời câu hỏi dựa trên tài liệu được cung cấp.

Quy tắc:
- Chỉ trả lời dựa trên CONTEXT bên dưới, không dùng kiến thức bên ngoài.
- Luôn trích dẫn số trang ở cuối mỗi ý, ví dụ: (Trang 12).
- Nếu không tìm thấy thông tin, hãy nói: "Tài liệu không đề cập đến vấn đề này."
- Trả lời bằng tiếng Việt, rõ ràng và súc tích.

CONTEXT:
{context}

CÂU HỎI: {query}

TRẢ LỜI:"""

def rag(query: str, top_k: int = TOP_K) -> dict:
    chunks   = retrieve(query, top_k=top_k)
    prompt   = build_prompt(query, chunks)
    response = llm.generate_content(prompt)
    return {
        "question": query,
        "answer":   response.text,
        "sources":  [{"page": c["page"], "score": c["score"]} for c in chunks],
    }

if __name__ == "__main__":
    questions = [
        "Quản trị tài chính là gì?",
        "Chương 1 nói về vấn đề gì?",
        "Định nghĩa khái niệm quản trị trong tài liệu?",
    ]
    for q in questions:
        print(f"\n{'='*60}")
        print(f"Câu hỏi: {q}")
        print(f"{'='*60}")
        result = rag(q)
        print(result["answer"])
        print("\nNguồn tham khảo:")
        for src in result["sources"]:
            print(f"  - Trang {src['page']} (score: {src['score']})")