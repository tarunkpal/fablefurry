import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.schema.runnable import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq # Using Groq for LLM
from streamlit_lottie import st_lottie
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import shutil
import glob


from dotenv import load_dotenv
load_dotenv()

SQLite fix for Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

#--helper functions--
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def format_docs(docs):
    d = "\n\n".join(doc for doc in docs)
    return str(d)

def build_vectorstore_from_stories():
    """Build vectorstore from story files if ChromaDB doesn't exist or is corrupted"""
    try:
        # Check if we have story files
        if not os.path.exists(MY_STORIES_DIR):
            st.error(f"Stories directory '{MY_STORIES_DIR}' not found!")
            return None
            
        story_files = glob.glob(os.path.join(MY_STORIES_DIR, "*.txt"))
        if not story_files:
            st.error(f"No .txt files found in '{MY_STORIES_DIR}'")
            return None
        
        # st.info(f"Building knowledge base from {len(story_files)} story files...")
        
        # Read all story content
        documents = []
        for filename in os.listdir(MY_STORIES_DIR):
            if filename.endswith(".txt"):
                file_path = os.path.join(MY_STORIES_DIR, filename)
                loader = TextLoader(file_path, encoding='utf-8', autodetect_encoding=True)
                try:
                    documents.extend(loader.load())
                except Exception as e:
                    print(f"  Warning: Could not load {filename}: {e}. Skipping this file.")

        
        if not documents:
            st.error("No valid story content found!")
            return None
        
        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} text chunks.")

        vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DB_DIR)
        
        # Persist the vectorstore
        vectorstore.persist()
        st.success(f"Knowledge base created successfully with {len(documents)} documents!")
        return vectorstore
        
    except Exception as e:
        st.error(f"Error building vectorstore: {e}")
        return None

lottie_url = "https://assets10.lottiefiles.com/packages/lf20_puciaact.json"
lottie_animation = load_lottieurl(lottie_url)

# --- Configuration ---
GROQ_API_KEY = os.getenv("api_key") or st.secrets.get("api_key", "")  # Support for Streamlit secrets
GROQ_MODEL_NAME = "llama3-8b-8192"

# --- Constants for RAG ---
MY_STORIES_DIR = "stories/" # Folder containing your story .txt files
CHROMA_DB_DIR = "./chroma_db" # Directory where the ChromaDB will be stored/loaded from
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Must be the same as used for building the DB

# --- Streamlit App ---
st.set_page_config(page_title="Fable Flurry", layout="centered")

st.title("✨ Fable Flurry - Magical Story Weaver for Kids ✨")
if lottie_animation:
    st_lottie(lottie_animation, speed=1, height=300, key="storybook")

st.markdown("Enter an idea, and I'll spin a unique story for children!")

# --- Knowledge Base Loading (Cached) ---
@st.cache_resource
def get_vectorstore():
    """
    Loads or creates the ChromaDB vectorstore with cloud deployment compatibility.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        
        # Check if ChromaDB exists and is valid
        if os.path.exists(CHROMA_DB_DIR) and os.listdir(CHROMA_DB_DIR):
            try:
                # Try to load existing ChromaDB
                vectorstore = Chroma(
                    persist_directory=CHROMA_DB_DIR, 
                    embedding_function=embeddings,
                    collection_name="stories"
                )
                
                # Test the vectorstore
                # test_results = vectorstore.similarity_search("test", k=1)
                st.success("Existing knowledge base loaded successfully!")
                return vectorstore
                
            except Exception as load_error:
                st.warning(f"Could not load existing database: {load_error}")
                st.info("Attempting to rebuild knowledge base...")
                
                # Remove corrupted database
                # if os.path.exists(CHROMA_DB_DIR):
                #     shutil.rmtree(CHROMA_DB_DIR)
        
        # Build new vectorstore
        return build_vectorstore_from_stories()
        
    except Exception as e:
        st.error(f"Error initializing vectorstore: {e}")
        return None

# --- LLM Initialization (Cached) ---
@st.cache_resource
def get_llm():
    """Initializes and returns the Groq LLM."""
    if not GROQ_API_KEY:
        st.error("Groq API key not found. Please set it in Streamlit secrets or environment variables.")
        st.info("For Streamlit Cloud, add your API key to the app's secrets.")
        return None
    try:
        llm_instance = ChatGroq(
            temperature=0.8,
            groq_api_key=GROQ_API_KEY,
            model_name=GROQ_MODEL_NAME
        )
        return llm_instance
    except Exception as e:
        st.error(f"Could not load Groq model '{GROQ_MODEL_NAME}'. Error: {e}")
        return None

# Initialize components
with st.spinner("Loading AI components..."):
    llm = get_llm()
    vectorstore = get_vectorstore()
    print(type(vectorstore))

# --- Main App Logic ---
if llm and vectorstore:
    print("hello")
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5}) # Reduced for cloud performance

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
        {
            "context": RunnableLambda(format_docs) | retriever, 
            "topic": RunnableLambda(lambda x: x["topic"])
        }
        | story_prompt
        | llm
    )

    # --- User Input for Story Idea ---
    st.header("Tell me your story idea! ✍️")
    user_story_idea = st.text_input(
        "What kind of story do you want? ",
        placeholder="A brave little mouse who saves the day..."
    )

    if st.button("✨ Generate Story", type="primary"):
        if user_story_idea.strip():
            with st.spinner("Spinning a magical tale for you..."):
                try:
                    generated_story = rag_chain.invoke({"topic": user_story_idea})
                    
                    st.subheader("📖 Here's your story!")
                    st.markdown("---")
                    
                    # Display the story with nice formatting
                    story_content = generated_story.content #if hasattr(generated_story, 'content') else str(generated_story)
                    st.write(story_content)
                    
                    st.markdown("---")
                    st.success("🎉 Story complete! Hope you enjoyed it!")
                    
                except Exception as e:
                    st.error(f"Failed to generate story: {e}")
                    st.info("Please try again. If the problem persists, check your internet connection.")
        else:
            st.warning("Please enter a story idea first!")

elif not llm:
    st.error("❌ AI model not available")
    st.info("Please check your Groq API key configuration.")
    
elif not vectorstore:
    st.error("❌ Knowledge base not available")
    st.info("Please ensure your story files are in the 'stories/' directory.")

else:
    st.info("⚙️ Setting up your story generator...")

# --- Sidebar with information ---
with st.sidebar:
    st.header("📚 About Fable Flurry")
    st.write("This app creates personalized children's stories using AI and a knowledge base of existing stories.")
    
    if st.button("🔄 Rebuild Knowledge Base"):
        # Clear cache and rebuild
        st.cache_resource.clear()
        if os.path.exists(CHROMA_DB_DIR):
            shutil.rmtree(CHROMA_DB_DIR)
        st.experimental_rerun()

st.markdown("---")
st.markdown("Powered by LangChain, Streamlit, and your wonderful stories!")
