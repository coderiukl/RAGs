import tempfile, os
import streamlit as st
from generator import rag
from chat_history import ChatHistory
from indexer import extract_text_from_pdf, split_into_chunks, embed_chunks, save_to_chromadb

st.set_page_config(
    page_title="RAG Chat",
    page_icon="📚",
    layout='wide'
)

st.title("RAG Chat - Hỏi đáp tài liệu PDF")

if "history" not in st.session_state:
    st.session_state.history = ChatHistory(max_turns=5)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed" not in st.session_state:
    st.session_state.indexed = False

with st.sidebar:
    st.header("Tài liệu")

    uploaded_file = st.file_uploader(
        "Upload file PDF",
        type=["pdf"],
        help="Hỗ trợ tiếng Việt"
    )

    if uploaded_file and not st.session_state.indexed:
        if st.button("Bắt đầu Index", type="primary", use_container_width=True):
            with st.spinner("Đang xử lý PDF..."):
                # Lưu file tạm
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                try:
                    # Chạy indexing pipeline
                    progress = st.progress(0, text="Đọc PDF...")
                    pages = extract_text_from_pdf(tmp_path)

                    progress.progress(30, text="Chia chunks...")
                    chunks = split_into_chunks(pages)

                    progress.progress(50, text=f"Embedding {len(chunks)} chunks...")
                    embeddings = embed_chunks(chunks)

                    progress.progress(85, text="Lưu vào ChromaDB...")
                    save_to_chromadb(chunks, embeddings)

                    progress.progress(100, text="Hoàn tất!")
                    st.session_state.indexed = True
                    st.session_state.pdf_name = uploaded_file.name
                    st.success(f"✓ Index xong {len(chunks)} chunks")

                finally:
                    os.unlink(tmp_path)  # xóa file tạm

    if st.session_state.indexed:
        st.success(f"✓ {st.session_state.get('pdf_name', 'PDF')} đã sẵn sàng")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Xóa lịch sử", use_container_width=True):
                st.session_state.history.clear()
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Đổi tài liệu", use_container_width=True):
                st.session_state.indexed = False
                st.session_state.history.clear()
                st.session_state.messages = []
                st.rerun()

    st.divider()
    st.caption(f"Memory: {len(st.session_state.history) // 2} lượt hội thoại")
    top_k = st.slider("Top-K chunks", 3, 15, 5)
    min_score = st.slider("Min score", 0.1, 0.9, 0.4, step=0.05)


# ── Khu vực chat ──────────────────────────────────────────
# Hiển thị lịch sử chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Nguồn tham khảo"):
                for src in msg["sources"]:
                    st.caption(f"Trang {src['page']} — score: {src['score']}")

# Input câu hỏi
if not st.session_state.indexed:
    st.info("Upload và index PDF ở sidebar để bắt đầu hỏi đáp.")
else:
    if query := st.chat_input("Nhập câu hỏi..."):
        # Hiện câu hỏi user
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Sinh câu trả lời
        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm và trả lời..."):
                result = rag(query, st.session_state.history, top_k=top_k)

            st.markdown(result["answer"])

            # Hiện nguồn tham khảo
            with st.expander("Nguồn tham khảo"):
                for src in result["sources"]:
                    st.caption(f"Trang {src['page']} — score: {src['score']}")

                if result["question"] != result["rephrased"]:
                    st.caption(f"Câu hỏi đã rephrase: _{result['rephrased']}_")

        # Lưu vào messages UI
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"],
        })