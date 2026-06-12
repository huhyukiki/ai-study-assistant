from __future__ import annotations

import hashlib
import os

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from streamlit.errors import StreamlitSecretNotFoundError

from src.chatbot import answer_question, generate_quiz
from src.document import chunk_pages, extract_pdf_pages
from src.retrieval import VectorIndex, create_embeddings


load_dotenv()

st.set_page_config(
    page_title="AI Study Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("AI Study Assistant")
st.caption("上传 PDF，基于资料提问，并生成带页码来源的回答与复习题。")


def get_setting(name: str, default: str = "") -> str:
    try:
        return str(st.secrets[name])
    except (KeyError, StreamlitSecretNotFoundError):
        return os.getenv(name, default)


with st.sidebar:
    st.header("设置")
    api_key = st.text_input(
        "OpenAI API Key",
        value=get_setting("OPENAI_API_KEY"),
        type="password",
        help="密钥只在当前应用进程中用于请求 OpenAI API。",
    )
    chat_model = st.text_input(
        "对话模型",
        value=get_setting("OPENAI_CHAT_MODEL", "gpt-4.1-mini"),
    )
    embedding_model = st.text_input(
        "向量模型",
        value=get_setting("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
    )
    top_k = st.slider("检索片段数", min_value=2, max_value=8, value=4)

uploaded_file = st.file_uploader("上传课程资料", type=["pdf"])

if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file is not None:
    pdf_bytes = uploaded_file.getvalue()
    document_key = hashlib.sha256(pdf_bytes).hexdigest()

    if st.session_state.get("document_key") != document_key:
        if not api_key:
            st.warning("请先在左侧输入 OpenAI API Key。")
            st.stop()

        with st.spinner("正在读取并索引 PDF..."):
            pages = extract_pdf_pages(pdf_bytes)
            chunks = chunk_pages(pages)
            if not chunks:
                st.error("没有从 PDF 中提取到文字。扫描版 PDF 需要先进行 OCR。")
                st.stop()

            client = OpenAI(api_key=api_key)
            embeddings = create_embeddings(
                client,
                (chunk.text for chunk in chunks),
                embedding_model,
            )
            st.session_state.index = VectorIndex(chunks, embeddings)
            st.session_state.document_key = document_key
            st.session_state.messages = []

        st.success(f"已索引 {len(pages)} 页，共 {len(chunks)} 个文本片段。")

    client = OpenAI(api_key=api_key)
    index: VectorIndex = st.session_state.index

    question_tab, quiz_tab = st.tabs(["资料问答", "生成复习题"])

    with question_tab:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("问一个关于资料的问题")
        if question:
            st.session_state.messages.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("正在检索资料并组织回答..."):
                    query_vector = create_embeddings(
                        client,
                        [question],
                        embedding_model,
                    )[0]
                    results = index.search(query_vector, top_k=top_k)
                    answer = answer_question(client, question, results, chat_model)
                st.markdown(answer)

                with st.expander("查看检索到的原文"):
                    for result in results:
                        st.markdown(
                            f"**第 {result.chunk.page} 页 · "
                            f"相似度 {result.score:.3f}**"
                        )
                        st.write(result.chunk.text)

            st.session_state.messages.append(
                {"role": "assistant", "content": answer}
            )

    with quiz_tab:
        st.write("从整份资料中检索代表性内容，生成一组复习题。")
        if st.button("生成 5 道复习题", type="primary"):
            with st.spinner("正在生成复习题..."):
                query_vector = create_embeddings(
                    client,
                    ["这份资料最重要的核心概念、事实和学习目标"],
                    embedding_model,
                )[0]
                results = index.search(query_vector, top_k=min(8, len(index.chunks)))
                quiz = generate_quiz(client, results, chat_model)
            st.markdown(quiz)
else:
    st.info("请上传一份包含可复制文字的 PDF 开始使用。")
