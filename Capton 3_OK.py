import os
import io
import re
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
import openai
from streamlit_webrtc import webrtc_streamer, AudioProcessorBase, WebRtcMode

# --- Load environment ---
load_dotenv()
QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
openai.api_key = OPENAI_API_KEY

# --- Inisialisasi LLM & embeddings ---
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
collection_name = "imdb_movies"

# ============================================================== #
# Load CSV & upload ke Qdrant
# ============================================================== #
@st.cache_data
def load_and_upload_csv_to_qdrant():
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
        st.success("✅ Koleksi 'imdb_movies' berhasil diunggah!")
    except Exception as e:
        st.warning(f"⚠️ Koleksi mungkin sudah ada. Melanjutkan. ({str(e)})")

load_and_upload_csv_to_qdrant()

# Gunakan koleksi yang sudah ada
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# ============================================================== #
# Tool & fungsi chatbot
# ============================================================== #
@tool
def get_relevant_docs(question: str):
    """
    Cari dokumen film relevan dari Qdrant berdasarkan pertanyaan user.

    Args:
        question (str): Judul film atau pertanyaan tentang film.

    Returns:
        list: List dokumen terkait dari Qdrant.
    """
    results = qdrant.similarity_search(question, k=5)
    return results

tools = [get_relevant_docs]

# ============================================================== #
# Fungsi rekomendasi film unik
# ============================================================== #
def get_similar_movies(title, top_k=3):
    try:
        similar_docs = qdrant.similarity_search(title, k=top_k + 50)
        def normalize_text(text): return re.sub(r'[^a-z0-9 ]', '', text.lower().strip())
        title_norm = normalize_text(title)
        unique_titles = set()
        recommendations = []
        for doc in similar_docs:
            candidate = doc.metadata.get("Series_Title", "")
            candidate_norm = normalize_text(candidate)
            if candidate_norm != title_norm and candidate_norm not in unique_titles:
                unique_titles.add(candidate_norm)
                recommendations.append(candidate)
            if len(recommendations) >= top_k: break
        while len(recommendations) < top_k:
            recommendations.append("(Belum cukup data relevan)")
        return recommendations[:top_k]
    except Exception as e:
        return [f"Error saat mencari rekomendasi: {str(e)}"]

# ============================================================== #
# Fungsi chatbot
# ============================================================== #
def chat_imdb(question):
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt="You are a movie expert. Use the tools to answer accurately about movies, genres, plots, directors, and stars."
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content

# ============================================================== #
# Streamlit UI
# ============================================================== #
st.set_page_config(page_title="🎬 Movie Master", page_icon="🎥", layout="wide")

with st.sidebar:
    st.title("🎬 Movie Lovers")
    st.markdown("🤖 **Your AI Movie Expert!**")
    st.markdown("Cari tahu sinopsis, pemeran, dan film serupa 🎞️")
    st.divider()
    st.markdown("**Made by:** Beby Hanzian\n**Powered by:** LangChain + Qdrant + Streamlit")

st.title("🎥 Movie Master Chatbot")

# ========================================================== #
# Rekam audio langsung dari browser (streamlit-webrtc)
# ========================================================== #
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_bytes = b""

    def recv(self, frame):
        self.audio_bytes += frame.to_ndarray().tobytes()
        return frame

webrtc_ctx = webrtc_streamer(
    key="mic",
    mode=WebRtcMode.SENDONLY,
    audio_processor_factory=AudioProcessor,
    media_stream_constraints={"audio": True, "video": False},
    async_processing=True
)

prompt_text = None
if st.button("🎤 Kirim ke Whisper") and webrtc_ctx.audio_processor:
    audio_processor = webrtc_ctx.audio_processor
    if audio_processor.audio_bytes:
        audio_file = io.BytesIO(audio_processor.audio_bytes)
        transcription = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        prompt_text = transcription.text
        st.markdown(f"**Transkrip:** {prompt_text}")

# ========================================================== #
# Input teks alternatif
# ========================================================== #
text_input = st.text_input("💬 Atau ketik pertanyaan tentang film...", "")
if text_input:
    prompt_text = text_input

# ========================================================== #
# Proses chatbot + rekomendasi
# ========================================================== #
if prompt_text:
    st.session_state.setdefault("messages", [])
    st.session_state.messages.append({"role": "Human", "content": prompt_text})

    with st.chat_message("Human", avatar="🧑‍💻"):
        st.markdown(prompt_text)

    with st.chat_message("AI", avatar="🎬"):
        with st.spinner("🎞️ Searching the movie database..."):
            answer = chat_imdb(prompt_text)
            st.markdown(answer)
            st.session_state.messages.append({"role": "AI", "content": answer})

            st.markdown("---")
            st.subheader("🎬 Rekomendasi Film Serupa:")
            recommendations = get_similar_movies(prompt_text)
            for idx, rec in enumerate(recommendations, start=1):
                st.markdown(f"{idx}. **{rec}**")
