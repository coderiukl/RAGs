import google.generativeai as genai
from retriever import retrieve, format_context
from config import GEMINI_API_KEY, GEMINI_MODEL, TOP_K
from chat_history import ChatHistory

genai.configure(api_key=GEMINI_API_KEY)
llm = genai.GenerativeModel(GEMINI_MODEL)

def rephrase_query(query: str, history: ChatHistory) -> str:
    if len(history) == 0:
        return query
    
    prompt = f"""Dựa vào lịch sử hội thoại bên dưới, hãy viết lại câu hỏi cuối
thành một câu hỏi độc lập, đầy đủ ngữ cảnh, không cần đọc lịch sử vẫn hiểu được.

Chỉ trả về câu hỏi đã viết lại, không giải thích gì thêm.

LỊCH SỬ HỘI THOẠI:
{history.format_for_prompt}

CÂU HỎI ĐỘC LẬP:"""
    
    response = llm.generate_content(prompt)
    rephrased = response.text.strip()
    print(f"[Rephrased] '{query}' -> '{rephrased}'")
    return rephrased

def build_prompt(query: str, chunks: list[dict], history: ChatHistory) -> str:

    history_text = history.format_for_prompt()
    history_section = f"\nLỊCH SỬ HỘI THOẠI:\n{history_text}\n" if history_text else ""

    context = format_context(chunks)
    return f"""Bạn là trợ lý AI chuyên trả lời câu hỏi dựa trên tài liệu được cung cấp.

Quy tắc:
- Chỉ trả lời dựa trên CONTEXT bên dưới, không dùng kiến thức bên ngoài.
- Luôn trích dẫn số trang ở cuối mỗi ý, ví dụ: (Trang 12).
- Nếu không tìm thấy thông tin, hãy nói: "Tài liệu không đề cập đến vấn đề này."
- Trả lời bằng tiếng Việt, rõ ràng và súc tích.
{history_section}
CONTEXT:
{context}

CÂU HỎI: {query}

TRẢ LỜI:"""

def rag(query: str, history: ChatHistory, top_k: int = TOP_K):
    search_query = rephrase_query(query, history)
    chunks   = retrieve(query, top_k=top_k)
    prompt   = build_prompt(query, chunks, history)
    response = llm.generate_content(prompt, stream=True)
    full_answer  = []
    for chunk in response:
        text = chunk.text
        full_answer += text
        yield text

    history.add("user", query)
    history.add("assistant", full_answer)

    yield {
        "question": query,
        "rephrased": search_query,
        "answer": full_answer,
        "sources":  [{"page": c["page"], "score": c["score"], 'source': c['source']} for c in chunks],
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