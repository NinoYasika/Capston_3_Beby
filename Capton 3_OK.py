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

# --- Load environment ---
load_dotenv()

# --- Ambil API key ---
QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# ============================================================== #
# 🧩 Inisialisasi model dan Qdrant di main thread
# ============================================================== #
@st.cache_resource
def init_qdrant_collection():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
    llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    collection_name = "imdb_movies"

    csv_path = os.path.join(os.path.dirname(__file__), "imdb_movies.csv")
    if not os.path.exists(csv_path):
        st.error("❌ File imdb_movies.csv tidak ditemukan.")
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
    except Exception:
        pass  # Koleksi mungkin sudah ada

    qdrant = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name=collection_name,
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY
    )

    return llm, embeddings, qdrant

# ============================================================== #
# 🎬 Fungsi rekomendasi film
# ============================================================== #
def get_similar_movies(qdrant, title, top_k=5):
    """Rekomendasi film berdasarkan kemiripan konten + genre"""
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

            doc_genres = [g.strip().lower() for g in doc.metadata.get("Genre", "").split(",")]
            genre_match = any(g in input_genres for g in doc_genres)

            unique_titles.add(movie_title)
            recommendations.append((genre_match, doc))

        recommendations.sort(key=lambda x: x[0], reverse=True)
        return [doc for _, doc in recommendations[:top_k]]

    except Exception as e:
        st.error(f"⚠️ Error mencari film serupa: {e}")
        return []

def show_movie_recommendations(qdrant, title, top_k=5):
    recommendations = get_similar_movies(qdrant, title, top_k=top_k)
    if not recommendations:
        st.info("🎬 Tidak ada film serupa.")
        return

    st.subheader("🎬 Rekomendasi Film Serupa:")
    for i, doc in enumerate(recommendations, start=1):
        rec_title = doc.metadata.get("Series_Title", "")
        genre = doc.metadata.get("Genre", "Unknown")
        overview = doc.metadata.get("Overview", "")
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
            st.markdown(f"Overview: {overview[:200]}{'...' if len(overview) > 200 else ''}")
        st.markdown("---")

# ============================================================== #
# 💬 Fungsi chatbot
# ============================================================== #
def chat_imdb(llm, qdrant, prompt):
    @tool
    def get_relevant_docs(question):
        return qdrant.similarity_search(question, k=5)

    tools = [get_relevant_docs]

    agent = create_react_agent(
        model=llm, tools=tools,
        prompt="You are a movie expert. Use the tools to answer accurately about movies."
    )

    result = agent.invoke({"messages": [{"role": "user", "content": prompt}]})
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

# ============================================================== #
# 🎨 Streamlit UI
# ============================================================== #
def main():
    st.set_page_config(page_title="🎬 Movie Master", page_icon="🎥", layout="wide")
    llm, embeddings, qdrant = init_qdrant_collection()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    st.title("🎥 Movie Master Chatbot")
    for msg in st.session_state.messages:
        avatar = "🧑‍💻" if msg["role"] == "Human" else "🎬"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])

    prompt = st.chat_input("Tanyakan sesuatu tentang film... 🎞️")
    if prompt:
        st.session_state.messages.append({"role": "Human", "content": prompt})
        with st.chat_message("Human", avatar="🧑‍💻"):
            st.markdown(prompt)

        with st.chat_message("AI", avatar="🎬"):
            with st.spinner("🎞️ Searching the movie database..."):
                response = chat_imdb(llm, qdrant, prompt)
                st.markdown(response["answer"])
                st.session_state.messages.append({"role": "AI", "content": response["answer"]})

                # Tampilkan rekomendasi film
                show_movie_recommendations(qdrant, prompt, top_k=5)

        with st.expander("📊 Token Usage & Tool Logs"):
            st.write(f"Input tokens: {response['total_input_tokens']}")
            st.write(f"Output tokens: {response['total_output_tokens']}")
            st.write(f"Estimated cost: Rp {response['price']:.4f}")
            st.code(response["tool_messages"])

if __name__ == "__main__":
    main()
