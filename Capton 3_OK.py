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

# ==============================================================
# 🔐 Load environment & API Keys
# ==============================================================
load_dotenv()

QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# ==============================================================
# ⚙️ Inisialisasi model dan embedding
# ==============================================================
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
collection_name = "imdb_movies"

# ==============================================================
# 📦 Upload CSV ke Qdrant (sekali saja, cached)
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
    except Exception as e:
        st.warning(f"⚠️ Koleksi mungkin sudah ada: {str(e)}")

load_and_upload_csv_to_qdrant()

qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# ==============================================================
# 🧠 Tool pencarian dokumen film
# ==============================================================
@tool
def get_relevant_docs(question):
    """Gunakan tool ini untuk mencari dokumen film terkait."""
    return qdrant.similarity_search(question, k=5)

tools = [get_relevant_docs]

# ==============================================================
# 🎬 Fungsi Rekomendasi Film
# ==============================================================
def get_similar_movies_by_features(title, top_k=3):
    try:
        search_results = qdrant.similarity_search(title, k=200)

        def normalize_text(t):
            return re.sub(r'[^a-z0-9 ]', '', str(t).lower().strip())

        title_norm = normalize_text(title)
        input_movie = None

        for doc in search_results:
            if normalize_text(doc.metadata.get("Series_Title", "")) == title_norm:
                input_movie = doc.metadata
                break

        if not input_movie:
            input_movie = search_results[0].metadata

        input_overview = normalize_text(input_movie.get("Overview", ""))
        input_rating = float(input_movie.get("IMDB_Rating", 0) or 0)
        input_cert = normalize_text(input_movie.get("Certificate", ""))
        input_genre = normalize_text(input_movie.get("Genre", ""))
        input_stars = {
            normalize_text(input_movie.get("Star1", "")),
            normalize_text(input_movie.get("Star2", "")),
            normalize_text(input_movie.get("Star3", "")),
            normalize_text(input_movie.get("Star4", "")),
        }

        unique_titles = set()
        scored_recommendations = []

        for doc in search_results:
            raw_title = doc.metadata.get("Series_Title", "")
            movie_title_norm = normalize_text(raw_title)
            if movie_title_norm == title_norm or movie_title_norm in unique_titles:
                continue

            overview = normalize_text(doc.metadata.get("Overview", ""))
            rating = float(doc.metadata.get("IMDB_Rating", 0) or 0)
            cert = normalize_text(doc.metadata.get("Certificate", ""))
            genre = normalize_text(doc.metadata.get("Genre", ""))
            stars = {
                normalize_text(doc.metadata.get("Star1", "")),
                normalize_text(doc.metadata.get("Star2", "")),
                normalize_text(doc.metadata.get("Star3", "")),
                normalize_text(doc.metadata.get("Star4", "")),
            }

            overlap_words = len(set(input_overview.split()) & set(overview.split()))
            overview_score = overlap_words / (len(set(input_overview.split())) + 1)
            rating_score = max(0, 1 - abs(float(rating) - float(input_rating)) / 5)
            cert_score = 0.5 if cert == input_cert and cert else 0.0
            genre_overlap = len(set(input_genre.split(",")) & set(genre.split(",")))
            genre_score = genre_overlap / max(1, len(set(input_genre.split(","))))
            star_overlap = len(input_stars & stars)
            star_score = star_overlap / 4.0

            total_score = (
                overview_score * 0.3
                + rating_score * 0.2
                + cert_score * 0.1
                + genre_score * 0.25
                + star_score * 0.15
            )

            scored_recommendations.append((doc, total_score))
            unique_titles.add(movie_title_norm)

        scored_recommendations.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in scored_recommendations[:top_k]]

    except Exception as e:
        st.error(f"❌ Terjadi kesalahan saat mencari rekomendasi: {e}")
        return []

# ==============================================================
# 💬 Chatbot dengan history
# ==============================================================
def chat_imdb(question, history):
    agent = create_react_agent(
        model=llm, tools=tools,
        prompt="You are a movie expert. Use the tools to answer accurately about movies."
    )

    # Kirim semua riwayat sebelumnya ke LLM
    messages = [{"role": msg["role"].lower(), "content": msg["content"]} for msg in history]
    messages.append({"role": "user", "content": question})

    result = agent.invoke({"messages": messages})
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
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages
    }

# ==============================================================
# 🎨 Tampilan Streamlit
# ==============================================================
st.set_page_config(page_title="🎬 Movie Lovers", page_icon="🎥", layout="wide")

with st.sidebar:
    st.title("🎬 Movie Lovers")
    st.markdown("🤖 **Your AI Movie Expert!**")
    st.divider()
    st.markdown("**Made by:** Beby Hanzian\n**Powered by:** LangChain + Qdrant + Streamlit + OpenAI")
    if st.button("🧹 Hapus Riwayat Chat"):
        st.session_state.messages = []
        st.rerun()

st.title("🎥 Movie Lovers")

current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir, "Movie Master Agent", "header_img.png")
if os.path.exists(image_path):
    st.image(image_path, width=800)

# ==============================================================
# 🕒 Inisialisasi & tampilkan history
# ==============================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

# Tampilkan semua percakapan lama
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "Human" else "🎬"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

# ==============================================================
# ✍️ Input chat baru
# ==============================================================
if prompt := st.chat_input("Tanyakan sesuatu tentang film... 🎞️"):
    # Tambah pesan user ke history
    st.session_state.messages.append({"role": "Human", "content": prompt})

    # Tampilkan langsung di UI
    with st.chat_message("Human", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Dapatkan jawaban dari model
    with st.chat_message("AI", avatar="🎬"):
        with st.spinner("🎞️ Searching the movie database..."):
            response = chat_imdb(prompt, st.session_state.messages)
            st.markdown(response["answer"])
            st.session_state.messages.append({"role": "AI", "content": response["answer"]})
            show_movie_recommendations(prompt, top_k=3)

    # Tampilkan token usage
    with st.expander("📊 Token Usage & Tool Logs"):
        st.write(f"Input tokens: {response['total_input_tokens']}")
        st.write(f"Output tokens: {response['total_output_tokens']}")
        st.write(f"Estimated cost: Rp {response['price']:.4f}")
        st.code(response["tool_messages"])
