

import streamlit as st
from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate

CHROMA_DIR = "./chroma_db"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "mistral"
TOP_K = 4


SYSTEM_PROMPT = """You are a knowledgeable and friendly marketing assistant for NovaSpark Electronics.
Use ONLY the context provided below to answer the customer's question.
If the answer is not in the context, say "I don't have that information in our current catalog —
please contact support at 1-800-NOVASPARK."
Be concise, helpful, and highlight key product benefits when relevant.

Context:
{context}

Chat History:
{chat_history}

Customer Question: {question}

Your Answer:"""


@st.cache_resource(show_spinner="Loading AI models...")
def load_chain():
    if not Path(CHROMA_DIR).exists():
        st.error(
            "Vector store not found. Run `python ingest.py` first to index the documents."
        )
        st.stop()

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
    )
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": TOP_K, "fetch_k": TOP_K * 3},
    )

    llm = Ollama(model=OLLAMA_MODEL, temperature=0.2)

    prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=SYSTEM_PROMPT,
    )

    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=False,
        output_key="answer",
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True,
        verbose=False,
    )
    return chain


def format_sources(source_docs) -> str:
    seen = set()
    lines = []
    for doc in source_docs:
        src = Path(doc.metadata.get("source", "unknown")).name
        if src not in seen:
            seen.add(src)
            lines.append(f"- {src}")
    return "\n".join(lines) if lines else "- (no source)"


# ─── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="NovaSpark Marketing Assistant",
    page_icon="⚡",
    layout="centered",
)

st.title("⚡ NovaSpark Marketing Assistant")
st.caption(
    "Ask me anything about our products, pricing, warranty, or latest news. "
    "All answers are grounded in our official catalog and press releases."
)

# ─── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("About")
    st.markdown(
        "**Model:** Mistral 7B (via Ollama — runs locally)\n\n"
        "**Embeddings:** all-MiniLM-L6-v2\n\n"
        "**Vector DB:** ChromaDB\n\n"
        "**Framework:** LangChain"
    )
    st.divider()
    st.subheader("Try asking:")
    example_questions = [
        "What is the price of NovaPhone X12?",
        "Which products have 2-year warranty?",
        "Tell me about NovaBuds Ultra noise cancellation",
        "What bundles are available?",
        "What was NovaSpark's Q1 2025 revenue?",
        "Does NovaWatch S3 work with Android?",
        "What smart home standards does NovaSpark Hub support?",
    ]
    for q in example_questions:
        st.markdown(f"• {q}")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ─── Chat state ─────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("Sources", expanded=False):
                st.markdown(msg["sources"])

# ─── Input & response ───────────────────────────────────────────────────────────

if user_input := st.chat_input("Ask about products, pricing, news..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get answer from RAG chain
    chain = load_chain()
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = chain.invoke({"question": user_input})
                answer = result["answer"]
                sources = format_sources(result.get("source_documents", []))
            except Exception as e:
                answer = (
                    f"Sorry, I encountered an error: {e}\n\n"
                    "Make sure Ollama is running (`ollama serve`) and "
                    f"the **{OLLAMA_MODEL}** model is pulled (`ollama pull {OLLAMA_MODEL}`)."
                )
                sources = ""

        st.markdown(answer)
        if sources:
            with st.expander("Sources", expanded=False):
                st.markdown(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )
