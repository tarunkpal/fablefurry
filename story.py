import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.schema.runnable import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq # Using Groq for LLM
from streamlit_lottie import st_lottie
import requests

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from dotenv import load_dotenv
load_dotenv()

#--helper functions--
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def format_docs(docs):
        d= "\n\n".join(doc for doc in docs)
        return str(d)

lottie_url = "https://assets10.lottiefiles.com/packages/lf20_puciaact.json"
lottie_animation = load_lottieurl(lottie_url)
# --- Configuration ---
GROQ_API_KEY = os.getenv("api_key") # Load from environment variable
GROQ_MODEL_NAME = "llama3-8b-8192"

# --- Constants for RAG ---
MY_STORIES_DIR = "my_stories/" # Folder containing your story .txt files
CHROMA_DB_DIR = "./chroma_db" # Directory where the ChromaDB will be stored/loaded from
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Must be the same as used for building the DB

# --- Streamlit App ---

st.set_page_config(page_title="Fable Flurry", layout="centered")

st.title("✨ Fable Flurry - Magical Story Weaver for Kids ✨")
st_lottie(lottie_animation, speed=1, height=300, key="storybook")


st.markdown("Enter an idea, and I'll spin a unique story for children!")

# --- Knowledge Base Loading (Cached) ---
@st.cache_resource
def get_vectorstore():
    """
    Loads the ChromaDB vectorstore from disk.
    This function will be cached by Streamlit.
    """
    if not os.path.exists(CHROMA_DB_DIR) or not os.listdir(CHROMA_DB_DIR):
        st.error(f"Knowledge base not found at '{CHROMA_DB_DIR}'.")
        st.info("Please run `python build_chroma_db.py` first to create the knowledge base.")
        return None

    # st.info("Loading existing knowledge base (ChromaDB) from disk...")
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings)
        # st.success("Knowledge base loaded successfully!")
        return vectorstore
    except Exception as e:
        st.error(f"Error loading knowledge base: {e}")
        st.info("The knowledge base might be corrupted or built with a different embedding model. Try deleting the 'chroma_db' folder and re-running `build_chroma_db.py`.")
        return None

# --- LLM Initialization (Cached) ---
@st.cache_resource
def get_llm():
    """Initializes and returns the Groq LLM."""
    if not GROQ_API_KEY:
        st.error("Groq API key not found. Please set the GROQ_API_KEY environment variable.")
        return None
    try:
        llm_instance = ChatGroq(
            temperature=0.8,
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL_NAME
        )
        # st.success(f"Groq model '{GROQ_MODEL_NAME}' loaded.")
        return llm_instance
    except Exception as e:
        st.error(f"Could not load Groq model '{GROQ_MODEL_NAME}'. Check your API key and internet connection. Error: {e}")
        return None

# Attempt to get the vectorstore and LLM
llm = get_llm()
vectorstore = get_vectorstore()

# --- Main App Logic ---
if llm and vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 7}) # Retrieve 7 relevant chunks
    print(type(retriever))

    # Define the RAG prompt
    story_template = """
    You are a friendly and imaginative children's storyteller for kids aged 5-8.
    Your stories are engaging, positive, easy to understand, and always have a clear beginning, middle, and a happy, resolved ending.
    Use descriptive but simple language. Keep sentences relatively short.
    The story should be between 500 and 800 words long. Avoid overly complex vocabulary or abstract concepts.
    Focus on a main character and their journey or problem-solving, introducing a clear problem and its resolution.

    Here are some examples of children's story passages, themes, and descriptive styles from a collection of existing stories. Use these as inspiration for your writing style, tone, and narrative structure:
    {context}

    Please write a full, original story based on the following idea: {topic}
    """
    story_prompt = ChatPromptTemplate.from_template(story_template)

    # Create the RAG chain
   

    rag_chain = (
        {"context": format_docs | retriever, "topic": lambda x: x["topic"]}
        | story_prompt
        | llm
    )

    # --- User Input for Story Idea ---
    st.header("Tell me your story idea! ✍️")
    user_story_idea = st.text_input(
        "What kind of story do you want? ",
        placeholder="Enter your story idea here..."
    )

    if st.button("Generate Story"):
        if user_story_idea:
            with st.spinner("Spinning a magical tale for you..."):
                try:
                    generated_story = rag_chain.invoke({"topic": user_story_idea})
                    st.subheader("Here's your story!")
                    st.write(generated_story.content) # Access .content for ChatGroq's output
                except Exception as e:
                    st.error(f"Failed to generate story: {e}")
                    st.info("Please check your Groq API key and internet connection. Also, ensure the knowledge base is loaded correctly.")
        else:
            st.warning("Please enter a story idea first!")
else:
    st.info("Story generation is disabled due to setup issues. Please check the error messages above. Make sure to run `python build_chroma_db.py` first!")

st.markdown("---")
st.markdown("Powered by LangChain, Streamlit, and your wonderful stories!")
