
import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# === Paths ===
FOLDER_PATH = r"C:\Users\Tiltomancer\Downloads\Lily\Legislation\Text"

@st.cache_resource
def load_legal_docs(folder_path):
    docs = []
    if not os.path.exists(folder_path):
        st.error(f"❌ The folder at {folder_path} doesn't exist!")
        return docs

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, file), encoding='utf-8')
            docs.extend(loader.load())
    return docs

@st.cache_resource
def create_vectorstore(_docs):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(_docs, embedding)
    return db

def is_relevant_question(question: str) -> bool:
    keywords = [
        "coastal", "rewilding", "restoration", "biodiversity", "wetland", "habitat",
        "birds directive", "habitats directive", "water framework", "marine",
        "climate law", "common agricultural policy", "nature restoration",
        "eu biodiversity strategy", "ecosystem", "NRL", "MSFD", "WFD", "CAP", "Floods"
    ]
    question_lower = question.lower()
    return any(kw in question_lower for kw in keywords)

# === Streamlit App ===
st.title("⚖️ EU Coastal Rewilding Legal Assistant (GMAL-Focused)")

with st.spinner("🔄 Loading EU legal documents..."):
    documents = load_legal_docs(FOLDER_PATH)
    vectordb = create_vectorstore(documents)

retriever = vectordb.as_retriever()
llm = Ollama(model="mistral")  # Replace with your model, e.g., "llama3"

# === Corrected Prompt Template ===
custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=""" 
You are an ecology and EU environmental law expert assisting GMAL (Green Marine Atlantic Landscapes) in planning coastal and estuarine ecosystem rewilding.

You must answer questions using the 8 EU directives related to restoration and rewilding (e.g. NRL, MSFD, WFD, CAP, Climate Law, Birds, Habitats, Floods).

Use only the following context from legal documents:
{context}

Question: {question}

Answer:
"""
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",
    chain_type_kwargs={"prompt": custom_prompt},
    return_source_documents=True
)

# === Query UI ===
query = st.text_input("🔍 Ask a question about the 8 coastal EU legislations:")

# Example buttons
col1, col2 = st.columns(2)
with col1:
    if st.button("💡 What habitats must be restored by 2030?"):
        query = "What habitats must be restored by 2030 under EU law?"
with col2:
    if st.button("🌿 How does the NRL help rewild saltmarshes?"):
        query = "How does the Nature Restoration Regulation help rewild saltmarshes?"

# Handle query
if query:
    if not is_relevant_question(query):
        st.warning("❌ Sorry, your question does not relate to the 8 EU legislations on coastal rewilding.")
    else:
        with st.spinner("💡 Processing your question..."):
            result = qa_chain(query)
            st.subheader("🧠 Answer")
            st.write(result['result'])

            st.subheader("📎 Source Documents")
            for doc in result['source_documents']:
                filename = os.path.basename(doc.metadata['source'])
                st.markdown(f"**Source:** `{filename}`")
                snippet = doc.page_content[:500].replace('\n', ' ') + "..."
                st.code(snippet)
else:
    st.info("💬 Enter a question or use a sample button to explore EU coastal rewilding laws.")
