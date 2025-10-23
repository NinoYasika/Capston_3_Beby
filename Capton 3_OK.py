import os
import re
import subprocess
import sys
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from difflib import get_close_matches
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ==============================================================
# Pastikan scikit-learn terinstal
# ==============================================================
subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn"], check=False)


# ==============================================================
# Load environment variables
# ==============================================================
load_dotenv()

QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
collection_name = "imdb_movies"


# ==============================================================
# Load CSV dan upload ke Qdrant
# ==============================================================
@st.cache_data
def load_and_upload_csv_to_qdrant():
    csv_path = os.path.join(os.path.dirname(__file__), "imdb_movies.csv")
    if not os.path.exists(csv_path):
        st.error("❌ File imdb_movies.csv tidak ditemukan.")
        st.stop()

    df = pd.read_csv(csv_path)
    df["combined_text"] = (
        "Title: " + df["Series_Title"].fillna("") + ". "
        + "Genre: " + df["Genre"].fillna("") + ". "
        + "Overview: " + df["Overview"].fillna("") + ". "
        + "Director: " + df["Director"].fillna("") + ". "
        + "Stars: " + df["Star1"].fillna("") + ", "
        + df["Star2"].fillna("") + ", "
        + df["Star3"].fillna("") + ", "
        + df["Star4"].fillna("")
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
    embedding=embeddings,
    collection_name=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


# ==============================================================
# Tool pencarian dokumen
# ==============================================================
@tool
def get_relevant_docs(question: str) -> list:
    """Mencari dokumen film terkait berdasarkan pertanyaan pengguna."""
    return qdrant.similarity_search(question, k=5)

tools = [get_relevant_docs]


# ==============================================================
# Fungsi rekomendasi film multi-kriteria (versi perbaikan)
# ==============================================================
def get_similar_movies_advanced(title: str, top_k: int = 3) -> list:
    """
    Rekomendasi film berdasarkan Genre, IMDb Rating, Certificate, dan kemiripan cerita.
    Menggunakan fuzzy matching agar lebih toleran pada variasi judul.
    """
    try:
        similar_docs = qdrant.similarity_search(title, k=80)
        if not similar_docs:
            return []

        def normalize_text(t):
            return re.sub(r'[^a-z0-9 ]', '', str(t).lower().strip())

        all_titles = [normalize_text(doc.metadata.get("Series_Title", "")) for doc in similar_docs]
        title_norm = normalize_text(title)
        closest_match = get_close_matches(title_norm, all_titles, n=1, cutoff=0.5)

        if not closest_match:
            input_doc = similar_docs[0]
        else:
            match_title = closest_match[0]
            input_doc = next(
                (doc for doc in similar_docs if normalize_text(doc.metadata.get("Series_Title", "")) == match_title),
                similar_docs[0]
            )

        input_genres = [g.strip().lower() for g in input_doc.metadata.get("Genre", "").split(",")]
        input_overview = input_doc.metadata.get("Overview", "")

        scored_docs = []

        for doc in similar_docs:
            if normalize_text(doc.metadata.get("Series_Title", "")) == normalize_text(
                    input_doc.metadata.get("Series_Title", "")):
                continue

            # Genre similarity
            doc_genres = [g.strip().lower() for g in doc.metadata.get("Genre", "").split(",")]
            genre_score = len(set(input_genres) & set(doc_genres)) / max(len(set(input_genres)), 1)

            # IMDb rating
            try:
                imdb_score = float(doc.metadata.get("IMDB_Rating", 0))
            except:
                imdb_score = 0
            imdb_score_norm = imdb_score / 10

            # Certificate similarity
            cert_score = 1 if doc.metadata.get("Certificate") == input_doc.metadata.get("Certificate") else 0

            # Overview similarity (TF-IDF cosine)
            tfidf = TfidfVectorizer(stop_words="english").fit([input_overview, doc.metadata.get("Overview", "")])
            overview_score = cosine_similarity(
                tfidf.transform([input_overview]),
                tfidf.transform([doc.metadata.get("Overview", "")])
            )[0][0]

            total_score = (
                0.4 * genre_score +
                0.3 * imdb_score_norm +
                0.1 * cert_score +
                0.2 * overview_score
            )

            scored_docs.append((total_score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)
        top_docs = [doc for score, doc in scored_docs[:top_k]]
        return top_docs

    except Exception as e:
        st.error(f"Error recommendation: {str(e)}")
        return []


# ==============================================================
# Menampilkan rekomendasi di UI
# ==============================================================
def show_movie_recommendations_advanced(title: str, top_k: int = 3):
    recommendations = get_similar_movies_advanced(title, top_k=top_k)
    if not recommendations:
        st.info("🎬 Tidak ada film serupa yang cocok.")
        return

    st.subheader("🎬 Rekomendasi Film Serupa:")
    for i, doc in enumerate(recommendations, start=1):
        rec_title = doc.metadata.get("Series_Title", "")
        genre = doc.metadata.get("Genre", "Unknown")
        imdb_rating = doc.metadata.get("IMDB_Rating", "N/A")
        certificate = doc.metadata.get("Certificate", "N/A")
        poster_url = doc.metadata.get("Poster_Link", "")

        cols = st.columns([1, 3])
        with cols[0]:
            if poster_url:
                st.image(poster_url, width=100)
            else:
                st.write("No poster available")
        with cols[1]:
            st.markdown(f"**{i}. {rec_title}**")
            st.markdown(f"Genre: {genre}")
            st.markdown(f"IMDb Rating: {imdb_rating}")
            st.markdown(f"Certificate: {certificate}")
        st.markdown("---")


# ==============================================================
# Chatbot utama
# ==============================================================
def chat_imdb(question: str, history: list) -> dict:
    agent = create_react_agent(
        model=llm,
        tools=tools,
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
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages
    }


# ==============================================================
# Streamlit Layout
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

            # Tampilkan rekomendasi multi-kriteria
            show_movie_recommendations_advanced(prompt, top_k=3)

    with st.expander("📊 Token Usage & Tool Logs"):
        st.write(f"Input tokens: {response['total_input_tokens']}")
        st.write(f"Output tokens: {response['total_output_tokens']}")
        st.write(f"Estimated cost: Rp {response['price']:.4f}")
        st.code(response["tool_messages"])
