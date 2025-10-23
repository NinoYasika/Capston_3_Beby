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

# ============================================================== #
# 🧩 Membaca CSV dan Upload ke Qdrant
# ============================================================== #

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

# ============================================================== #
# 🎬 Tool dan fungsi chatbot
# ============================================================== #

@tool
def get_relevant_docs(question):
    """Gunakan tool ini untuk mencari dokumen film terkait."""
    return qdrant.similarity_search(question, k=5)

tools = [get_relevant_docs]

# ============================================================== #
# 🧠 Fungsi rekomendasi film berdasarkan genre
# ============================================================== #

def get_similar_movies_by_genre(title, top_k=3):
    try:
        similar_docs = qdrant.similarity_search(title, k=50)

        def normalize_text(t):
            return re.sub(r'[^a-z0-9 ]', '', t.lower().strip())

        title_norm = normalize_text(title)
        input_genre = ""
        for doc in similar_docs:
            if normalize_text(doc.metadata.get("Series_Title", "")) == title_norm:
                input_genre = doc.metadata.get("Genre", "")
                break

        input_genres = [g.strip().lower() for g in input_genre.split(",")]

        unique_titles = set()
        recommendations = []

        for doc in similar_docs:
            raw_title = doc.metadata.get("Series_Title", "")
            movie_title = normalize_text(raw_title)
            if movie_title == title_norm or movie_title in unique_titles:
                continue

            doc_genre = doc.metadata.get("Genre", "")
            doc_genres = [g.strip().lower() for g in doc_genre.split(",")]
            if any(g in input_genres for g in doc_genres):
                unique_titles.add(movie_title)
                recommendations.append(doc)

            if len(recommendations) >= top_k:
                break

        return recommendations  # langsung kembalikan list dokumen

    except Exception as e:
        return []

def show_movie_recommendations(title, top_k=3):
    recommendations = get_similar_movies_by_genre(title, top_k=top_k)

    if not recommendations:
        st.info("🎬 Tidak ada film serupa berdasarkan genre.")
        return

    st.subheader("🎬 Rekomendasi Film Serupa:")
    for i, doc in enumerate(recommendations, start=1):
        rec_title = doc.metadata.get("Series_Title", "")
        genre = doc.metadata.get("Genre", "Unknown")
        poster_url = doc.metadata.get("Poster_Link", "")

        cols = st.columns([1, 3])
        with cols[0]:
            if poster_url:
                st.image(poster_url, width=100)
            else:
                st.write("No poster")
        with cols[1]:
            st.markdown(f"**{i}. {rec_title}**")
            st.markdown(f"Genre: {genre}")
        st.markdown("---")

# ============================================================== #
# 💬 Fungsi utama chatbot
# ============================================================== #

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

# ============================================================== #
# 🎨 Tampilan Streamlit
# ============================================================== #

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

            # Tampilkan rekomendasi film visual berdasarkan genre
            show_movie_recommendations(prompt, top_k=3)

    with st.expander("📊 Token Usage & Tool Logs"):
        st.write(f"Input tokens: {response['total_input_tokens']}")
        st.write(f"Output tokens: {response['total_output_tokens']}")
        st.write(f"Estimated cost: Rp {response['price']:.4f}")
        st.code(response["tool_messages"])
