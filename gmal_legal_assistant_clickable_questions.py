
import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# === API Key from Streamlit Secrets ===
os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]

# === UI ===
st.set_page_config(page_title="GMAL Legal Assistant", page_icon="⚖️")
st.image("logo.png", width=150)
st.title("⚖️ GMAL EU Legal Assistant (AI-powered)")

# === Sidebar: Directive List ===
with st.sidebar:
    st.markdown("### 📚 Directives Included")
    st.markdown("""
- Nature Restoration Regulation (EU) 2024/1991  
- EU Biodiversity Strategy for 2030  
- Habitats Directive (92/43/EEC)  
- Birds Directive (2009/147/EC)  
- Water Framework Directive (2000/60/EC)  
- Marine Strategy Framework Directive (2008/56/EC)  
- Floods Directive (2007/60/EC)  
- Common Agricultural Policy (CAP) (2023–2027)  
- European Climate Law (2021/1119)
    """)

# === Suggested Questions (Clickable) ===
st.markdown("#### 💡 Example Questions")
example_questions = [
    "What would a rewilded Atlantic coast look like in 2030?",
    "How does the Nature Restoration Regulation support wetland recovery?",
    "Which EU laws guide urban rewilding under GMAL?",
    "What role do rivers and estuaries play in the 2030 scenario?",
    "How does the EU Biodiversity Strategy align with climate adaptation?",
    "What are the key restoration targets in the Birds and Habitats Directives?",
    "How do marine protected areas link to the Marine Strategy Framework Directive?",
    "Which indicators must be met by 2027 under the Water Framework Directive?",
    "How can we use CAP funding for rewilding agricultural land?",
]

selected_example = None
for i, q in enumerate(example_questions):
    if st.button(q, key=f"ex_{i}"):
        selected_example = q

# === Paths ===
FOLDER_PATH = "Text"

@st.cache_resource
def load_legal_docs(folder_path):
    docs = []
    if not os.path.exists(folder_path):
        st.error(f"❌ The folder at {folder_path} doesn't exist!")
        return docs

    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            loader = TextLoader(os.path.join(folder_path, file), encoding="utf-8")
            docs.extend(loader.load())
    return docs

@st.cache_resource
def create_vectorstore(_docs):
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_documents(_docs, embedding)
    return db

def is_relevant_question(question: str) -> bool:
    keywords = [
        "coastal", "rewilding", "restoration", "biodiversity", "wetland", "habitat",
        "birds directive", "habitats directive", "water framework", "marine",
        "climate law", "common agricultural policy", "nature restoration",
        "eu biodiversity strategy", "ecosystem", "NRL", "MSFD", "WFD", "CAP", "Floods",
        "estuaries", "saltmarsh", "dune", "peatland", "kelp", "marine protected area",
        "ecological integrity", "passive rewilding", "hydromorphology", "floodplain", 
        "green infrastructure", "urban nature", "tree canopy", "nature-based solution"
    ]
    question_lower = question.lower()
    return any(kw in question_lower for kw in keywords)

# === Load and Run ===
with st.spinner("🔄 Loading legal documents..."):
    documents = load_legal_docs(FOLDER_PATH)
    vectordb = create_vectorstore(documents)

retriever = vectordb.as_retriever()
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

custom_prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=""" 
You are an EU legal and ecological assistant supporting GMAL (Green Marine Atlantic Landscapes) in understanding EU directives related to rewilding and restoration.

Use the context below to answer clearly, citing directives and restoration timelines where appropriate.

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

# === Handle user question
query = st.text_input("🔍 Ask a question about EU coastal and restoration laws:")

# Override with example if selected
if selected_example:
    query = selected_example
    st.markdown(f"**Selected Question:** _{query}_")

if query:
    if not is_relevant_question(query):
        st.error("⚠️ This topic may fall outside the 9 EU coastal restoration directives used in GMAL. Please ask about rewilding, restoration, or the listed legislation.")
    else:
        with st.spinner("💡 Generating answer..."):
            result = qa_chain(query)
            st.subheader("🧠 Answer")
            st.write(result['result'])

            st.subheader("📎 Sources")
            for doc in result['source_documents']:
                st.markdown(f"**File:** `{os.path.basename(doc.metadata['source'])}`")
                st.code(doc.page_content[:500] + "...")
else:
    st.info("💬 Enter a question above or click a suggestion.")
