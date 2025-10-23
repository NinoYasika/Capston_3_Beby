import os
import re
import random
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from difflib import get_close_matches
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent

# =============== SETUP ENVIRONMENT ===============
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# =============== STREAMLIT PAGE CONFIG ===============
st.set_page_config(
    page_title="🎬 Smart Movie Recommender",
    page_icon="🎥",
    layout="wide"
)

# =============== HEADER IMAGE ===============
st.image("header_img.jpg", use_container_width=True)
st.title("🎬 Smart Movie Recommendation App")
st.markdown("Temukan film serupa berdasarkan genre, rating, dan alur cerita 🎞️")

# =============== CONNECT QDRANT & LLM ===============
embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)

qdrant = QdrantVectorStore.from_existing_collection(
    collection_name="movies_collection",
    embedding=embeddings,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

agent_executor = create_react_agent(llm, tools=[])


# =====================================================
# FUNCTION: ADVANCED MOVIE RECOMMENDATION
# =====================================================
def get_similar_movies_advanced(title: str, top_k: int = 3) -> list:
    """
    Rekomendasi film berdasarkan Genre, IMDb Rating, Certificate, dan kemiripan cerita.
    Versi FIXED: Hilangkan duplikat dan pastikan hasil unik.
    """
    try:
        similar_docs = qdrant.similarity_search(title, k=80)
        if not similar_docs:
            return []

        def normalize_text(t):
            return re.sub(r'[^a-z0-9 ]', '', str(t).lower().strip())

        # cari dokumen input yang paling mirip
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
        seen_titles = set()

        for doc in similar_docs:
            doc_title = doc.metadata.get("Series_Title", "")
            norm_title = normalize_text(doc_title)

            # lewati film duplikat dan film yang sama
            if norm_title in seen_titles or norm_title == normalize_text(input_doc.metadata.get("Series_Title", "")):
                continue
            seen_titles.add(norm_title)

            # genre score
            doc_genres = [g.strip().lower() for g in doc.metadata.get("Genre", "").split(",")]
            genre_score = len(set(input_genres) & set(doc_genres)) / max(len(set(input_genres)), 1)

            # IMDb score
            try:
                imdb_score = float(doc.metadata.get("IMDB_Rating", 0))
            except:
                imdb_score = 0
            imdb_score_norm = imdb_score / 10

            # certificate score
            cert_score = 1 if doc.metadata.get("Certificate") == input_doc.metadata.get("Certificate") else 0

            # overview similarity
            doc_overview = doc.metadata.get("Overview", "")
            if input_overview.strip() and doc_overview.strip():
                tfidf = TfidfVectorizer(stop_words="english").fit([input_overview, doc_overview])
                overview_score = cosine_similarity(
                    tfidf.transform([input_overview]),
                    tfidf.transform([doc_overview])
                )[0][0]
            else:
                overview_score = 0

            # total skor dengan sedikit random jitter untuk hasil unik
            total_score = (
                0.4 * genre_score +
                0.3 * imdb_score_norm +
                0.1 * cert_score +
                0.2 * overview_score +
                random.uniform(0.0001, 0.001)
            )

            scored_docs.append((total_score, doc))

        # urutkan
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # ambil hasil unik & terbaik
        unique_recs = []
        used_titles = set()
        for score, doc in scored_docs:
            t = normalize_text(doc.metadata.get("Series_Title", ""))
            if t not in used_titles:
                used_titles.add(t)
                unique_recs.append((score, doc))
            if len(unique_recs) >= top_k:
                break

        return unique_recs

    except Exception as e:
        st.error(f"Error recommendation: {str(e)}")
        return []


# =====================================================
# DISPLAY RECOMMENDATION
# =====================================================
def show_movie_recommendations_advanced(title: str, top_k: int = 3):
    recommendations = get_similar_movies_advanced(title, top_k=top_k)
    if not recommendations:
        st.info("🎬 Tidak ada film serupa yang cocok.")
        return

    st.subheader("🎬 Rekomendasi Film Serupa:")
    for i, (score, doc) in enumerate(recommendations, start=1):
        rec_title = doc.metadata.get("Series_Title", "")
        genre = doc.metadata.get("Genre", "Unknown")
        imdb_rating = doc.metadata.get("IMDB_Rating", "N/A")
        certificate = doc.metadata.get("Certificate", "N/A")
        poster_url = doc.metadata.get("Poster_Link", "")
        overview = doc.metadata.get("Overview", "")

        similarity_percent = round(score * 100, 1)

        cols = st.columns([1, 3])
        with cols[0]:
            if poster_url:
                st.image(poster_url, width=110)
            else:
                st.write("🎞️ No poster available")

        with cols[1]:
            st.markdown(f"**{i}. {rec_title}**  \n⭐ IMDb: {imdb_rating} | 🎫 {certificate}  \n🎭 Genre: {genre}")
            st.markdown(f"<small>🔹 Kemiripan: {similarity_percent}%</small>", unsafe_allow_html=True)
            st.caption(overview[:200] + "..." if overview else "")
        st.markdown("---")


# =====================================================
# MAIN APP
# =====================================================
st.markdown("### 🔍 Cari Film Favoritmu")
movie_input = st.text_input("Masukkan judul film (contoh: Inception, Titanic, Joker):")

if st.button("🎥 Tampilkan Rekomendasi"):
    if movie_input.strip():
        show_movie_recommendations_advanced(movie_input)
    else:
        st.warning("Masukkan judul film terlebih dahulu.")
