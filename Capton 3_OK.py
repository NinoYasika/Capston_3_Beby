# app.py
import os
import streamlit as st
from dotenv import load_dotenv
import requests
from typing import Optional
from html import escape

# --- Load .env (local dev only; Streamlit secrets take precedence) ---
load_dotenv()

# --- Secrets / API keys (Streamlit secrets recommended) ---
QDRANT_URL = st.secrets.get("QDRANT_URL", os.getenv("QDRANT_URL"))
QDRANT_API_KEY = st.secrets.get("QDRANT_API_KEY", os.getenv("QDRANT_API_KEY"))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
OMDB_API_KEY = st.secrets.get("OMDB_API_KEY", os.getenv("OMDB_API_KEY"))  # optional

# --- LangChain/OpenAI/Qdrant imports (kept as in your original) ---
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage

# --- Initialize LLM & embeddings (keep model choice as you prefer) ---
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

# --- Connect Qdrant (assumes collection exists) ---
collection_name = "imdb_movies"
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# --- Tool: fetch relevant docs from Qdrant ---
@tool
def get_relevant_docs(question: str):
    """Find similar movie documents from Qdrant."""
    results = qdrant.similarity_search(question, k=5)
    # return minimal serializable representation
    simple = []
    for r in results:
        # attempt to extract a title/metadata if present
        meta = getattr(r, "metadata", {}) or {}
        title = meta.get("title") or meta.get("name") or getattr(r, "page_content", "")[:120]
        simple.append({"title": title, "content": getattr(r, "page_content", "")[:800], "metadata": meta})
    return simple

tools = [get_relevant_docs]

# --- Helper: call OMDb for movie details (optional) ---
def get_movie_info_from_omdb(title: str) -> Optional[dict]:
    if not OMDB_API_KEY or not title:
        return None
    try:
        params = {"t": title, "apikey": OMDB_API_KEY}
        resp = requests.get("http://www.omdbapi.com/", params=params, timeout=8)
        data = resp.json()
        if data.get("Response") == "True":
            return {
                "title": data.get("Title"),
                "year": data.get("Year"),
                "poster": data.get("Poster") if data.get("Poster") and data.get("Poster") != "N/A" else None,
                "rating": data.get("imdbRating"),
                "genre": data.get("Genre"),
                "plot": data.get("Plot"),
                "director": data.get("Director"),
                "actors": data.get("Actors")
            }
    except Exception:
        return None
    return None

# --- Core: chat function using create_react_agent (like your original) ---
def chat_imdb(question: str, history: str):
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt="You are a movie expert and only answer questions about movies. Use tools to fetch movie details when needed."
    )
    result = agent.invoke({"messages": [{"role": "user", "content": question}]})
    # Extract answer (safely)
    answer = result["messages"][-1].content if result and "messages" in result else "Sorry, no response."

    # Compute token usage (supports multiple metadata formats)
    total_input_tokens = 0
    total_output_tokens = 0
    for message in result["messages"]:
        meta = getattr(message, "response_metadata", {}) or {}
        if "usage_metadata" in meta:
            um = meta["usage_metadata"]
            total_input_tokens += um.get("input_tokens", 0)
            total_output_tokens += um.get("output_tokens", 0)
        elif "token_usage" in meta:
            tu = meta["token_usage"]
            total_input_tokens += tu.get("prompt_tokens", 0)
            total_output_tokens += tu.get("completion_tokens", 0)

    # Estimate price (same formula you used; adjust rates as appropriate)
    price = 17_000 * (total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1_000_000

    # Extract tool messages for inspection
    tool_messages = []
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            tool_messages.append(message.content)

    return {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages
    }

# -------------------------
# UI / Streamlit layout
# -------------------------
st.set_page_config(page_title="Movie Master Chatbot", layout="wide", initial_sidebar_state="auto")

# CSS for chat bubbles + light/dark base
st.markdown(
    """
    <style>
    .chat-container { max-width: 900px; margin: 0 auto; }
    .user-bubble {
        background: linear-gradient(90deg,#e6fff2,#dcffd9);
        padding:12px; border-radius:12px; margin:6px 0; display:inline-block;
        box-shadow:0 1px 3px rgba(0,0,0,0.08);
    }
    .bot-bubble {
        background: linear-gradient(90deg,#f3f7ff,#e7efff);
        padding:12px; border-radius:12px; margin:6px 0; display:inline-block;
        box-shadow:0 1px 3px rgba(0,0,0,0.06);
    }
    .meta { color: #6b7280; font-size: 0.85rem; margin-top:4px; }
    .title-center { text-align:center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar controls: theme, filters, favorites
with st.sidebar:
    st.header("Controls")
    dark_mode = st.checkbox("Dark Mode (applies basic style)", value=False)
    if dark_mode:
        st.markdown(
            """
            <style>body{background-color:#0b1221;color:#e6eef8} .stApp {background-color:#0b1221}</style>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Search Filters")
    genre_filter = st.selectbox("Genre (hint for LLM)", ["", "Action", "Drama", "Comedy", "Horror", "Sci-Fi", "Romance", "Documentary"])
    year_from, year_to = st.slider("Year range", 1900, 2026, (1990, 2025))
    st.write("---")
    st.subheader("Saved")
    if "favorites" not in st.session_state:
        st.session_state.favorites = []
    for i, fav in enumerate(st.session_state.favorites):
        st.markdown(f"{i+1}. {escape(fav)}")

    if st.button("Clear Favorites"):
        st.session_state.favorites = []
        st.success("Favorites cleared")

# Header
st.markdown('<div class="title-center"><h1>🎬 Movie Master Chatbot</h1><p>Ask anything about movies — cast, trivia, plots, recommendations.</p></div>', unsafe_allow_html=True)

# Image/banner (if available locally), attempt to show robustly
current_dir = os.path.dirname(__file__)
banner = None
possible_paths = [
    os.path.join(current_dir, "Movie Master Agent", "header_img.png"),
    os.path.join(current_dir, "header_img.png")
]
for p in possible_paths:
    if os.path.exists(p):
        banner = p
        break
if banner:
    st.image(banner, use_column_width=True)

# Initialize session messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Chat input area (single-column main)
col1, col2 = st.columns([3, 1])
with col1:
    prompt = st.chat_input("Ask me any movie question (e.g., 'Tell me about Inception' or 'Recommend sci-fi movies 2010-2020')")

with col2:
    # quick actions
    if st.button("🎲 Random Movie Trivia"):
        # ask LLM for a quick fact
        trivia = chat_imdb("Give me one interesting movie trivia or behind-the-scenes fact.", "")
        st.session_state.messages.append({"role": "AI", "content": trivia["answer"]})

# Process input
if prompt:
    # stitch filters into question to guide LLM
    q_parts = [prompt]
    if genre_filter:
        q_parts.append(f"Focus on genre: {genre_filter}")
    if year_from or year_to:
        q_parts.append(f"Years between {year_from} and {year_to}")
    full_question = " | ".join(q_parts)

    # display and store user message
    st.session_state.messages.append({"role": "Human", "content": prompt})

    # show user bubble immediately
    st.write(f"<div class='user-bubble'>{escape(prompt)}</div>", unsafe_allow_html=True)

    # call chat function
    with st.spinner("Thinking..."):
        response = chat_imdb(full_question, "\n".join([m["content"] for m in st.session_state.messages[-20:]]))
    answer = response["answer"]

    # add to history and display as bot bubble
    st.session_state.messages.append({"role": "AI", "content": answer})
    st.write(f"<div class='bot-bubble'>{escape(answer).replace('\\n','<br/>')}</div>", unsafe_allow_html=True)

    # Try to detect a movie title to call OMDb (simple heuristic: if user asked 'about X' or had quotes)
    detected_title = None
    # naive heuristics
    if "about " in prompt.lower():
        detected_title = prompt.split("about ", 1)[1].strip().strip('"').strip("'")
    elif prompt.count('"') >= 2:
        detected_title = prompt.split('"')[1]
    # fallback: check Qdrant tool messages for titles
    if response.get("tool_messages"):
        # look for short titles in tool messages
        for t in response["tool_messages"]:
            if isinstance(t, str) and len(t) < 60:
                # choose shortest plausible
                if not detected_title or len(t) < len(detected_title):
                    detected_title = t

    movie_info = get_movie_info_from_omdb(detected_title) if detected_title else None

    # show OMDb info if found
    if movie_info:
        st.markdown("---")
        cols = st.columns([1, 3])
        with cols[0]:
            if movie_info.get("poster"):
                st.image(movie_info["poster"], width=180)
            else:
                st.info("Poster not available")
            # Save favorite button
            if st.button("❤️ Save Movie to Favorites", key=f"fav_{movie_info['title']}"):
                st.session_state.favorites.append(movie_info["title"])
                st.success(f"Saved: {movie_info['title']}")
        with cols[1]:
            st.markdown(f"### {movie_info['title']} ({movie_info['year']})")
            st.markdown(f"**IMDb:** {movie_info.get('rating', 'N/A')}  •  **Genre:** {movie_info.get('genre', 'N/A')}")
            st.markdown(f"**Director:** {movie_info.get('director', 'N/A')}")
            st.markdown(f"**Actors:** {movie_info.get('actors', 'N/A')}")
            st.markdown(f"**Plot:** {movie_info.get('plot', 'N/A')}")
        st.markdown("---")

    # Expanders for tools, usage
    with st.expander("🔎 Tool Calls (from vector DB):"):
        st.code(response.get("tool_messages", "No tool messages"))

    with st.expander("🕘 Chat History (last 20):"):
        st.code("\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.messages[-20:]]))

    with st.expander("📊 Usage & Estimated Cost:"):
        st.markdown(f"- Input tokens: {response['total_input_tokens']}\n- Output tokens: {response['total_output_tokens']}\n- Estimated price: Rp {response['price']:.2f}")

    # Optional TTS using gTTS (if installed)
    try:
        from gtts import gTTS
        from io import BytesIO
        if st.button("🔊 Listen (TTS)"):
            tts = gTTS(text=answer, lang="en", slow=False)
            mp3_fp = BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            st.audio(mp3_fp.read(), format="audio/mp3")
    except Exception:
        # gTTS not available; silently skip
        pass

# Render full conversation in chat-style (read-only) below input
st.markdown("## Conversation")
for message in st.session_state.messages[-50:]:
    role = message["role"]
    content = message["content"]
    if role == "Human":
        st.write(f"<div class='user-bubble'>{escape(content)}</div>", unsafe_allow_html=True)
    else:
        st.write(f"<div class='bot-bubble'>{escape(content).replace('\\n','<br/>')}</div>", unsafe_allow_html=True)

# Footer / quick tips
st.markdown("---")
st.markdown("**Tips:** Try queries like _Tell me about \"Inception\"_, _Recommend sci-fi movies 2010-2020_, or _Who played in The Godfather?_")
