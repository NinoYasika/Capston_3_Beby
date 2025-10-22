import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage
from dotenv import load_dotenv

# --- Load environment file ---
load_dotenv()

# --- Ambil API key dari secrets atau .env ---
QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

# --- Inisialisasi model ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key=OPENAI_API_KEY
)

# --- Inisialisasi Qdrant Vector Store ---
collection_name = "imdb_movies"
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# --- Tool untuk mencari dokumen film relevan ---
@tool
def get_relevant_docs(question):
    """Gunakan tool ini untuk mencari dokumen film terkait."""
    results = qdrant.similarity_search(question, k=5)
    return results

tools = [get_relevant_docs]

# --- Fungsi untuk menampilkan rekomendasi film serupa ---
def get_similar_movies(title, top_k=3):
    try:
        similar_docs = qdrant.similarity_search(title, k=top_k + 1)
        # Buang film yang sama dengan input utama
        filtered = [doc for doc in similar_docs if doc.page_content.lower() != title.lower()]
        recommendations = [doc.page_content for doc in filtered[:top_k]]
        if recommendations:
            return recommendations
        else:
            return ["Tidak ditemukan film mirip dalam database."]
    except Exception as e:
        return [f"Error saat mencari rekomendasi: {str(e)}"]

# --- Fungsi utama chatbot ---
def chat_imdb(question, history):
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt="You are a master of knowledge about movies. Answer only questions about movies using the provided tools."
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    answer = result["messages"][-1].content

    # Hitung token
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

    # Ambil pesan dari tool
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


# --- Tampilan Streamlit ---
st.set_page_config(page_title="🎬 Movie Master", page_icon="🎥", layout="wide")

# Sidebar info
with st.sidebar:
    st.title("🎬 Movie Master Bot")
    st.markdown("🤖 **Your AI Movie Expert!**")
    st.markdown("Cari tahu sinopsis, pemeran, rating IMDb, dan rekomendasi film serupa 🎞️")
    st.divider()
    st.markdown("**Made by:** Alfian\n**Powered by:** LangChain + Qdrant + Streamlit")

# Header
st.title("🎥 Movie Master Chatbot")
current_dir = os.path.dirname(__file__)
image_path = os.path.join(current_dir, "Movie Master Agent", "header_img.png")
st.image(image_path, use_container_width=True)

# --- Inisialisasi chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Tampilkan riwayat chat ---
for message in st.session_state.messages:
    avatar = "🧑‍💻" if message["role"] == "Human" else "🎬"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# --- Input pengguna ---
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

            # --- Rekomendasi film mirip ---
            st.markdown("---")
            st.subheader("🎬 Rekomendasi Film Serupa:")
            recommendations = get_similar_movies(prompt)
            for idx, rec in enumerate(recommendations, start=1):
                st.markdown(f"{idx}. **{rec}**")

    # Ekspander detail teknis
    with st.expander("📊 Token Usage & Tool Logs"):
        st.write(f"**Input tokens:** {response['total_input_tokens']}")
        st.write(f"**Output tokens:** {response['total_output_tokens']}")
        st.write(f"**Estimated cost:** Rp {response['price']:.4f}")
        st.write("**Tool Messages:**")
        st.code(response["tool_messages"])
