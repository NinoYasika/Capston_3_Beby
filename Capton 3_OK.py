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
        "Certificate: " + df["Certificate"].fillna("") + ". " +
        "Overview: " + df["Overview"].fillna("") + ". " +
        "Director: " + df["Director"].fillna("") + ". " +
        "Stars: " + df["Star1"].fillna("") + ", " + df["Star2"].fillna("") + ", " +
        df["Star3"].fillna("") + ", " + df["Star4"].fillna("") + ". " +
        "IMDb Rating: " + df["IMDB_Rating"].fillna("").astype(str)
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
    return qdrant.similarity_search(question, k=50)

tools = [get_relevant_docs]

# ============================================================== #
# 🧠 Fungsi rekomendasi film berdasarkan multi-kriteria
# ============================================================== #

def get_similar_movies(title_or_question, top_k=3):
    """
    Rekomendasi 3 film paling relevan berdasarkan:
    - Kemiripan cerita (overview)
    - Genre
    - Certificate
    - IMDb Rating
    Film utama tidak muncul.
    """
    try:
        docs = qdrant.similarity_search(title_or_question, k=50)
        if not docs:
            return []

        def normalize_title(title):
            return re.sub(r'\s+', ' ', title.strip().lower())

        input_norm = normalize_title(title_or_question)

        main_doc_title = docs[0].metadata.get("Series_Title", "")
        main_norm = normalize_title(main_doc_title)

        # Ranking berdasarkan multi-kriteria
        scored_docs = []
        for doc in docs:
            doc_title = doc.metadata.get("Series_Title", "")
            if not doc_title or normalize_title(doc_title) == main_norm:
                continue

            score = 0

            # Genre overlap
            genre_input = docs[0].metadata.get("Genre", "").lower().split(", ")
            genre_doc = doc.metadata.get("Genre", "").lower().split(", ")
            genre_score = len(set(genre_input) & set(genre_doc)) / max(len(set(genre_input)), 1)
            score += genre_score * 0.4

            # Certificate match
            cert_input = docs[0].metadata.get("Certificate", "").lower()
            cert_doc = doc.metadata.get("Certificate", "").lower()
            if cert_input and cert_input == cert_doc:
                score += 0.2

            # IMDb rating closeness
            try:
                rating_input = float(docs[0].metadata.get("IMDB_Rating", 0))
                rating_doc = float(doc.metadata.get("IMDB_Rating", 0))
                rating_diff = abs(rating_input - rating_doc)
                rating_score = max(0, 1 - rating_diff / 10)
                score += rating_score * 0.4
            except:
                pass

            scored_docs.append((score, doc))

        # Sort descending berdasarkan score
        scored_docs.sort(key=lambda x: x[0], reverse=True)

        # Ambil top_k
        recommendations = [doc for score, doc in scored_docs[:top_k]]
        return recommendations

    except Exception as e:
        st.error(f"❌ Error mencari rekomendasi: {e}")
        return []

def show_movie_recommendations(title_or_question, top_k=3):
    recommendations = get_similar_movies(title_or_question, top_k=top_k)
    if not recommendations:
        st.info("🎬 Tidak ada film serupa ditemukan.")
        return

    st.subheader("🎬 Rekomendasi Film Serupa:")
    for i, doc in enumerate(recommendations, start=1):
        rec_title = doc.metadata.get("Series_Title", "")
        year = doc.metadata.get("Released_Year", "Unknown")
        genre = doc.metadata.get("Genre", "Unknown")
        cert = doc.metadata.get("Certificate", "Unknown")
        rating = doc.metadata.get("IMDB_Rating", "Unknown")
        poster_url = doc.metadata.get("Poster_Link", "")

        cols = st.columns([1, 3])
        with cols[0]:
            if poster_url:
                st.image(poster_url, width=100)
            else:
                st.write("No poster")
        with cols[1]:
            st.markdown(f"**{i}. {rec_title} ({year})**")
            st.markdown(f"Genre: {genre} | Certificate: {cert} | IMDb: {rating}")
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

            # Tampilkan rekomendasi film visual berdasarkan multi-kriteria
            show_movie_recommendations(prompt, top_k=3)

    with st.expander("📊 Token Usage & Tool Logs"):
        st.write(f"Input tokens: {response['total_input_tokens']}")
        st.write(f"Output tokens: {response['total_output_tokens']}")
        st.write(f"Estimated cost: Rp {response['price']:.4f}")
        st.code(response["tool_messages"])
