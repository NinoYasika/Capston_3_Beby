import os
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
# 🧩 Membaca CSV dan Upload ke Qdrant (hanya pertama kali)
# ==============================================================

@st.cache_data
def load_and_upload_csv_to_qdrant():
    csv_path = os.path.join(os.path.dirname(__file__), "imdb_movies.csv")
    if not os.path.exists(csv_path):
        st.error("❌ File imdb_movies.csv tidak ditemukan di folder proyek.")
        st.stop()

    df = pd.read_csv(csv_path)

    # Gabungkan kolom penting menjadi satu teks untuk embedding
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
        qdrant_store = QdrantVectorStore.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            collection_name=collection_name,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        st.success("✅ Koleksi 'imdb_movies' berhasil diunggah ke Qdrant!")
    except Exception as e:
        st.warning(f"⚠️ Koleksi mungkin sudah ada. Melanjutkan dengan koleksi yang ada. ({str(e)})")

load_and_upload_csv_to_qdrant()

# --- Gunakan koleksi yang sudah ada ---
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# ==============================================================
# 🎬 Tool dan fungsi chatbot
# ==============================================================

@tool
def get_relevant_docs(question):
    """Gunakan tool ini untuk mencari dokumen film terkait."""
    results = qdrant.similarity_search(question, k=5)
    return results

tools = [get_relevant_docs]

def get_similar_movies(title, top_k=3):
    try:
        similar_docs = qdrant.similarity_search(title, k=top_k + 1)
        filtered = [doc for doc in similar_docs if title.lower() not in doc.page_content.lower()]
        recommendations = [doc.metadata["Series_Title"] for doc in filtered[:top_k]]
        if recommendations:
            return recommendations
        else:
            return ["Tidak ditemukan film mirip dalam database."]
    except Exception as e:
        return [f"Error saat mencari rekomendasi: {str(e)}"]

def chat_imdb(question, history):
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt="You are a movie expert. Use the tools to answer accurately about movies, genres, plots, directors, and stars."
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content

    total_input_tokens = 0
    total_output_tokens = 0
    for message in result["messages"]:
        if "usage_metadata" in message.response_metadata:
            total_input_tokens += message.response_metadata["usage_metadata"]["input_tokens"]
            total_output_tokens += message.response_metadata["usage_metadata"]["output_tokens"]
        elif "token_usage" in message.response_metadata:
            total_input_tokens += message.response_metadata["token_usage"].get("prompt_tokens", 0)
            total_output_tokens += message.response_metadata["token_usage"].get("completion_tokens", 0)

    price = 17000 * (total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1_000_000

    tool_messages = [
        message.content for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]

    return {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages
    }

# ==============================================================
# 💬 Tampilan Streamlit
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
    st.image(image_path, use_container_width=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "Human" else "🎬"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

if prompt := st.chat_input("Tanyakan sesuatu tentang film... 🎞️"):
    st.session_state.messages.append({"role": "Human", "content": prompt})
    with st.chat_message("Human", avatar="🧑‍💻"):
        st.markdown(prompt)

    with st.chat_message("AI", avatar="🎬"):
        with st.spinner("🎞️ Searching the movie database..."):
            response = chat_imdb(prompt, st.session_state.messages)
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "AI", "content": answer})

            st.markdown("---")
            st.subheader("🎬 Rekomendasi Film Serupa:")
            recommendations = get_similar_movies(prompt)
            for idx, rec in enumerate(recommendations, start=1):
                st.markdown(f"{idx}. **{rec}**")

    with st.expander("📊 Token Usage & Tool Logs"):
        st.write(f"**Input tokens:** {response['total_input_tokens']}")
        st.write(f"**Output tokens:** {response['total_output_tokens']}")
        st.write(f"**Estimated cost:** Rp {response['price']:.4f}")
        st.write("**Tool Messages:**")
        st.code(response["tool_messages"])
