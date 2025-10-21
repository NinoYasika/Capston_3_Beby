# import os
import streamlit as st

QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage

from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# --- Load environment file ---
load_dotenv()

# --- Ambil API key dari .env ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- Pastikan API key terbaca ---
print("API Key loaded:", OPENAI_API_KEY is not None)

# --- Inisialisasi model ---
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=OPENAI_API_KEY
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    api_key = OPENAI_API_KEY)

collection_name = "imdb_movies"
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY)

@tool
def get_relevant_docs(question):
  """Use this tools for get relevant documents about recipes."""
  results = qdrant.similarity_search(
      question,
      k=5
  )
  return results

tools = [get_relevant_docs]

def chat_imdb(question, history):
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt= f'''You are a master of knowledge about movies. Answer only question about movies and use given tools for get movies details.'''
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    answer = result["messages"][-1].content

    total_input_tokens = 0
    total_output_tokens = 0

    for message in result["messages"]:
        if "usage_metadata" in message.response_metadata:
            total_input_tokens += message.response_metadata["usage_metadata"]["input_tokens"]
            total_output_tokens += message.response_metadata["usage_metadata"]["output_tokens"]
        elif "token_usage" in message.response_metadata:
            # Fallback for older or different structures
            total_input_tokens += message.response_metadata["token_usage"].get("prompt_tokens", 0)
            total_output_tokens += message.response_metadata["token_usage"].get("completion_tokens", 0)

    price = 17_000*(total_input_tokens*0.15 + total_output_tokens*0.6)/1_000_000

    tool_messages = []
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            tool_message_content = message.content
            tool_messages.append(tool_message_content)

    response = {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages
    }
    return response

st.title("Chatbot Movies Master")
# st.image("C:/Users/Hi/Capstone 3/Movie Master Agent/header_img.png")

# Dapatkan lokasi file Python saat ini
current_dir = os.path.dirname(__file__)
# Gabungkan path menuju folder dan file gambar
image_path = os.path.join(current_dir, "Movie Master Agent", "header_img.png")

# Tampilkan gambar
st.image(image_path)


# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask me movies question"):
    messages_history = st.session_state.get("messages", [])[-20:]
    history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history]) or " "

    # Display user message in chat message container
    with st.chat_message("Human"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "Human", "content": prompt})
    
    # Display assistant response in chat message container
    with st.chat_message("AI"):
        response = chat_imdb(prompt, history)
        answer = response["answer"]
        st.markdown(answer)
        st.session_state.messages.append({"role": "AI", "content": answer})

    with st.expander("**Tool Calls:**"):
        st.code(response["tool_messages"])

    with st.expander("**History Chat:**"):
        st.code(history)

    with st.expander("**Usage Details:**"):
        st.code(f'input token : {response["total_input_tokens"]}\noutput token : {response["total_output_tokens"]}')



# Untuk menjalankan streamlit copi ini (tanpa #) di comand prompt:
# streamlit run "C:\Users\Hi\Capstone_Tiga\Capton 3_OK.py"



