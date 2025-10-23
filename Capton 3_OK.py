import os
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage

# --- Load environment file ---
load_dotenv()

# --- Ambil API key dari secrets atau .env ---
QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# --- Inisialisasi model dan embedding ---
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
collection_name = "imdb_movies"

# ==============================================================
# 🧩 Membaca CSV dan Upload ke Qdrant
# ==============================================================

@st.cache_data
def load_and_upload_csv_to_qdrant():
    csv_path = os.path.join(os.path.dirname(__file__), "imdb_movies.csv")
    if not os.path.exists(csv_path):
        st.error("❌ File imdb_movies.csv tidak ditemukan di folder proyek.")
        st.stop()

    df = pd.read_csv(csv_path)
    df["combined_text"] = (
        "Title: " + df["Series_Title"].fillna("") + ". " +
        "Genre: " + df["Genre"].fillna("") + ". " +
        "Overview: " + df["Overview"].fillna("") + ". " +
        "Director: " + df["Director"].fillna("") + ". " +
        "Stars: " + df["Star1"].fillna("") + ", " + df["Star2"].fillna("") + ", " +
        df["Star3"].fillna("") + ", " + df["Star4"].fillna("")
    )

    texts = df["combined_text"].tolist()
    metadatas = df.to_dict(orient="records")

    try:
        QdrantVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            collection_name=collection_name,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        st.success("✅ Koleksi 'imdb_movies' berhasil diunggah ke Qdrant!")
    except Exception as e:
        st.warning(f"⚠️ Koleksi mungkin sudah ada: {str(e)}")

load_and_upload_csv_to_qdrant()

qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings, collection_name=collection_name,
    url=QDRANT_URL, api_key=QDRANT_API_KEY
)

# ==============================================================
# 🎬 Tool dan fungsi chatbot
# ==============================================================

@tool
def get_relevant_docs(question):
    """Gunakan tool ini untuk mencari dokumen film terkait."""
    return qdrant.similarity_search(question, k=5)

tools = [get_relevant_docs]

# ==============================================================
# 🧠 Fungsi rekomendasi film unik (3 hasil berbeda)
# ==============================================================

def get_similar_movies(title, top_k=2):
    try:
        similar_docs = qdrant.similarity_search(title, k=top_k + 50)

        def normalize_text(t):
            return re.sub(r'[^a-z0-9 ]', '', t.lower().strip())

        title_norm = normalize_text(title)
        unique_titles = set()
        filtered = []

        for doc in similar_docs:
            raw_title = doc.metadata.get("Series_Title", "")
            movie_title = normalize_text(raw_title)
            if movie_title != title_norm and movie_title not in unique_titles:
                unique_titles.add(movie_title)
                filtered.append(doc)
            if len(filtered) >= top_k:
                break

        recommendations = [d.metadata["Series_Title"] for d in filtered[:top_k]]
        recommendations = [r for r in recommendations if normalize_text(r) != title_norm]

        # Tambah hasil cadangan jika masih kurang
        if len(recommendations) < top_k:
            for d in similar_docs:
                title_extra = normalize_text(d.metadata.get("Series_Title", ""))
                if title_extra not in unique_titles and title_extra != title_norm:
                    recommendations.append(d.metadata["Series_Title"])
                    unique_titles.add(title_extra)
                if len(recommendations) >= top_k:
                    break

        while len(recommendations) < top_k:
            recommendations.append("(Belum cukup data relevan)")

        return recommendations[:top_k]

    except Exception as e:
        return [f"Error: {str(e)}"]

# ==============================================================
# 💬 Fungsi utama chatbot
# ==============================================================

def chat_imdb(question, history):
    agent = create_react_agent(
        model=llm, tools=tools,
        prompt="You are a movie expert. Use the tools to answer accurately about movies."
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content

    total_input_tokens = sum(
        msg.response_metadata.get("usage_metadata", {}).get("input_tokens", 0)
        for msg in result["messages"]
    )
    total_output_tokens = sum(
        msg.response_metadata.get("usage_metadata", {}).get("output_tokens", 0)
        for msg in result["messages"]
    )

    price = 17000 * (total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1_000_000
    tool_messages = [msg.content for msg in result["messages"] if isinstance(msg, ToolMessage)]

    return {
        "answer": answer, "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages
    }

# ==============================================================
# 🎨 Tampilan Streamlit
# ==============================================================

st.set_page_config(page_title="🎬 Movie Master", page_icon="🎥", layout="wide")

with st.sidebar:
    st.title("🎬 Movie Lovers")
    st.markdown("🤖 **Your AI Movie Expert!**")
    st.markdown("Cari tahu sinopsis, pemeran, dan film serupa 🎞️")
    st.divider()
    st.markdown("**Made by:** Beby Hanzian\n**Powered by:** LangChain + Qdrant + Streamlit")

st.title("🎥 Movie Master Chatbot")

current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir, "Movie Master Agent", "header_img.png")
if os.path.exists(image_path):
    st.image(image_path, width=800)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "Human" else "🎬"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if prompt := st.chat_input("Tanyakan sesuatu tentang film... 🎞️"):
    st.session_state.messages.append({"role": "Human", "content": prompt})
    with st.chat_message("Human", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("AI", avatar="🎬"):
        with st.spinner("🎞️ Searching the movie database..."):
            response = chat_imdb(prompt, st.session_state.messages)
            st.markdown(response["answer"])
            st.session_state.messages.append({"role": "AI", "content": response["answer"]})

            st.markdown("---")
            st.subheader("🎬 Rekomendasi Film Serupa:")
            for i, rec in enumerate(get_similar_movies(prompt), start=1):
                st.markdown(f"{i}. **{rec}**")

    with st.expander("📊 Token Usage & Tool Logs"):
        st.write(f"Input tokens: {response['total_input_tokens']}")
        st.write(f"Output tokens: {response['total_output_tokens']}")
        st.write(f"Estimated cost: Rp {response['price']:.4f}")
        st.code(response["tool_messages"])
