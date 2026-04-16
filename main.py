from generator import rag
from chat_history import ChatHistory

history = ChatHistory(max_turns=5)

print("RAG system ready. Type 'quit' to exit.\n")
print(f"Type 'clear' to delete memory | 'quit' to exit\n")
while True:
    query = input("Question: ").strip()
    if query.lower() in ("quit", "exit", "q"):
        break
    
    if query.lower() == "clear":
        history.clear()
        print(f"Deleted this history.\n")
        continue

    if not query:
        continue

    result = rag(query, history)
    
    print(f"Assistant: {result['answer']}")
    print("\nNguồn")
    for src in result['sources']:
        print(f"Trang {src['page']} - score: {src['score']}")
    print(f"  [Đang nhớ {len(history) // 2} lượt hội thoại]\n")