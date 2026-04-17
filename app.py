import tempfile, os
import streamlit as st
from generator import rag
from chat_history import ChatHistory
from indexer import extract_text_from_pdf, split_into_chunks, embed_chunks, save_to_chromadb, index_multiple_pdfs

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

    uploaded_files = st.file_uploader(
        "Upload file PDF",
        type=["pdf"],
        accept_multiple_files=True,
        help="Hỗ trợ tiếng Việt"
    )

    if uploaded_files and not st.session_state.indexed:
        if st.button("Bắt đầu Index", type="primary", use_container_width=True):
            tmp_paths = []
            try:
                with st.spinner("Đang xử lý..."):
                    progress = st.progress(0, text="Chuẩn bị...")

                    # Lưu tất cả file tạm
                    for f in uploaded_files:
                        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                        tmp.write(f.read())
                        tmp.close()
                        tmp_paths.append((tmp.name, f.name))

                    all_chunks = []
                    for idx, (tmp_path, original_name) in enumerate(tmp_paths):
                        pct = int((idx / len(tmp_paths)) * 80)
                        progress.progress(pct, text=f"Đọc {original_name}...")

                        pages = extract_text_from_pdf(tmp_path)
                        # Ghi đè source = tên file gốc (không phải tên tmp)
                        for p in pages:
                            p["source"] = original_name
                        chunks = split_into_chunks(pages)
                        all_chunks.extend(chunks)

                    progress.progress(82, text=f"Embedding {len(all_chunks)} chunks...")
                    embeddings = embed_chunks(all_chunks)

                    progress.progress(95, text="Lưu vào ChromaDB...")
                    save_to_chromadb(all_chunks, embeddings)

                    progress.progress(100, text="Hoàn tất!")
                    st.session_state.indexed = True
                    st.session_state.indexed_files = [f.name for f in uploaded_files]
                    st.success(f"✓ Index xong {len(all_chunks)} chunks từ {len(uploaded_files)} file")

            finally:
                for tmp_path, _ in tmp_paths:
                    os.unlink(tmp_path)

    # Hiện danh sách file đã index
    if st.session_state.indexed:
        st.success("Sẵn sàng hỏi đáp")
        with st.expander(f"{len(st.session_state.indexed_files)} file đã index"):
            for name in st.session_state.indexed_files:
                st.caption(f"📄 {name}")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Xóa lịch sử", use_container_width=True):
                st.session_state.history.clear()
                st.session_state.messages = []
                st.rerun()
        with col2:
            if st.button("Đổi tài liệu", use_container_width=True):
                st.session_state.indexed = False
                st.session_state.indexed_files = []
                st.session_state.history.clear()
                st.session_state.messages = []
                st.rerun()

    st.divider()
    st.caption(f"Memory: {len(st.session_state.history) // 2} lượt hội thoại")
    top_k = st.slider("Top-K chunks", 3, 15, 5)
    min_score = st.slider("Min score", 0.1, 0.9, 0.4, step=0.05)

# ── Chat area ──────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Nguồn tham khảo"):
                for src in msg["sources"]:
                    st.caption(f"📄 {src['source']} — Trang {src['page']} (score: {src['score']})")

if not st.session_state.indexed:
    st.info("Upload và index PDF ở sidebar để bắt đầu.")
else:
    if query := st.chat_input("Nhập câu hỏi..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            place_holder = st.empty()
            full_text = ""
            metadata = None

            for chunk in rag(query, st.session_state.history, top_k=top_k):
                if isinstance(chunk, dict):
                    metadata = chunk
                else:
                    full_text += chunk
                    place_holder.markdown(full_text + "▌")

            place_holder.markdown(full_text)
            if metadata:
                with st.expander("Nguồn tham khảo"):
                    for src in metadata["sources"]:
                        st.caption(f"📄 {src['source']} — Trang {src['page']} (score: {src['score']})")
                    if metadata["question"] != metadata["rephrased"]:
                        st.caption(f"Rephrase: _{metadata['rephrased']}_")

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_text,
            "sources": metadata["sources"] if metadata else [],
        })