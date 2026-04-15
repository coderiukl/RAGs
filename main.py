from generator import rag

print("RAG system ready. Type 'quit' to exit.\n")
while True:
    query = input("Question: ").strip()
    if query.lower() in ("quit", "exit", "q"):
        break
    if not query:
        continue
    result = rag(query)
    print(f"\n{result['answer']}")
    print("\nNguồn")
    for src in result['sources']:
        print(f"Trang {src['page']} - score: {src['score']}")
    print()
