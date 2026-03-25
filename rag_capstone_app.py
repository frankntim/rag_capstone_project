# =========================
# app.py
# =========================

import streamlit as st
from typing import TypedDict, List
import concurrent.futures

# LangChain / LangGraph
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langgraph.graph import StateGraph, END

# Cross-encoder
from sentence_transformers import CrossEncoder

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Agentic RAG", layout="wide")

st.title("🤖 Agentic RAG (Streaming + Multi-Agent + Memory)")

# =========================
# SIDEBAR (TRACES)
# =========================
st.sidebar.title("🧭 Agent Traces")

if "traces" not in st.session_state:
    st.session_state.traces = []

def add_trace(msg):
    st.session_state.traces.append(msg)

with st.sidebar:
    for t in st.session_state.traces[-20:]:
        st.write(t)

# =========================
# CHAT MEMORY
# =========================
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================
# LOAD DATA (cached)
# =========================
@st.cache_resource
def load_vectorstore():
    with open("state_of_the_union.txt", "r") as f:
      text = f.read()

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([text])

    embedding = OpenAIEmbeddings()

    vs = Chroma.from_documents(
        docs,
        embedding=embedding,
        persist_directory="./chroma_db"
    )

    return vs

vectorstore = load_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 20})

# =========================
# CROSS-ENCODER
# =========================
@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

cross_encoder = load_reranker()

def rerank(query, docs, top_n=5):
    pairs = [[query, d.page_content] for d in docs]
    scores = cross_encoder.predict(pairs)
    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [doc for doc, _ in ranked[:top_n]]

# =========================
# TOOL
# =========================
@tool
def retrieve_context(query: str) -> str:
    docs = retriever.get_relevant_documents(query)
    reranked = rerank(query, docs, top_n=5)
    return "\n\n".join([d.page_content for d in reranked])

# =========================
# LLM (STREAMING)
# =========================
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

def stream_to_container(stream, container):
    full = ""
    for chunk in stream:
        if chunk.content:
            full += chunk.content
            container.markdown(full)
    return full

# =========================
# STATE
# =========================
class State(TypedDict):
    query: str
    context: str
    draft_answer: str
    critique: str
    final_answer: str
    history: List[str]
    iterations: int

# =========================
# AGENTS
# =========================

# 🔍 Retriever
def retriever_agent(state: State):
    add_trace("🔍 Retriever: fetching context")

    context = retrieve_context.invoke(state["query"])

    return {
        "context": context,
        "iterations": state.get("iterations", 0) + 1
    }

# 🧠 Generator (streaming UI)
def generator_agent(state: State):
    add_trace("🧠 Generator: generating answer")

    container = st.chat_message("assistant")
    placeholder = container.empty()

    history_text = "\n".join(state["history"])

    prompt = f"""
    Conversation History:
    {history_text}

    Context:
    {state['context']}

    Question:
    {state['query']}

    Answer conversationally and clearly.
    """

    stream = llm.stream(prompt)
    answer = stream_to_container(stream, placeholder)

    return {"draft_answer": answer}

# 🧐 Critic (runs in parallel)
def critic_agent_fn(query, context, answer):
    critique_prompt = f"""
    Evaluate answer.

    Question: {query}

    Context:
    {context}

    Answer:
    {answer}

    If correct → APPROVED else explain.
    """

    return llm.invoke(critique_prompt).content

# 🔀 Parallel Node (Retriever + Critic)
def parallel_node(state: State):
    add_trace("⚡ Parallel: retriever + critic")

    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_retriever = executor.submit(retriever_agent, state)

        # Wait for retriever first (critic needs context)
        retriever_result = future_retriever.result()

        future_critic = executor.submit(
            critic_agent_fn,
            state["query"],
            retriever_result["context"],
            state.get("draft_answer", "")
        )

        critique = future_critic.result()

    return {
        "context": retriever_result["context"],
        "critique": critique,
        "iterations": retriever_result["iterations"]
    }

# 🧭 Supervisor
def supervisor(state: State):
    critique = state["critique"]
    iterations = state["iterations"]

    add_trace(f"🧭 Supervisor iteration {iterations}")

    if "APPROVED" in critique:
        add_trace("✅ Approved")
        return "finalize"

    if iterations >= 3:
        add_trace("⚠️ Max iterations")
        return "finalize"

    add_trace("🔁 Retry")
    return "retry"

# 🎯 Final
def finalize(state: State):
    return {"final_answer": state["draft_answer"]}

# =========================
# GRAPH
# =========================
builder = StateGraph(State)

builder.add_node("parallel", parallel_node)
builder.add_node("generate", generator_agent)
builder.add_node("finalize", finalize)

builder.set_entry_point("parallel")

builder.add_edge("parallel", "generate")

builder.add_conditional_edges(
    "generate",
    lambda s: "finalize",  # generator always goes to supervisor via state
    {"finalize": "parallel"}  # dummy (we control via external loop)
)

# We will manually control loop instead of pure graph recursion
graph = builder.compile()

# =========================
# CHAT UI
# =========================
user_input = st.chat_input("Ask a question about the speech...")

if user_input:
    st.session_state.chat_history.append(f"User: {user_input}")

    st.chat_message("user").markdown(user_input)

    state = {
        "query": user_input,
        "context": "",
        "draft_answer": "",
        "critique": "",
        "final_answer": "",
        "history": st.session_state.chat_history,
        "iterations": 0
    }

    # 🔁 LOOP (Supervisor-controlled)
    for _ in range(3):
        state.update(parallel_node(state))
        state.update(generator_agent(state))

        critique = critic_agent_fn(
            state["query"],
            state["context"],
            state["draft_answer"]
        )
        state["critique"] = critique

        decision = supervisor(state)

        if decision == "finalize":
            break

    st.session_state.chat_history.append(
        f"Assistant: {state['draft_answer']}"
    )

    st.success("✅ Done")
