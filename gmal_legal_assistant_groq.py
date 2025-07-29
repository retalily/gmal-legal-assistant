
import os
import streamlit as st
import requests

# === Load Groq API Key from Secrets ===
GROQ_API_KEY = st.secrets["groq_api_key"]

# === UI Setup ===
st.set_page_config(page_title="GMAL Legal Assistant", page_icon="⚖️")
st.image("logo.png", width=150)
st.title("⚖️ GMAL EU Legal Assistant (Groq-powered)")

# === Sidebar Directives ===
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

# === Example Clickable Questions ===
st.markdown("#### 💡 Example Questions")
example_questions = [
    "What would a rewilded Atlantic coast look like in 2030?",
    "How does the Nature Restoration Regulation support wetland recovery?",
    "How do the Birds and Habitats Directives work together?",
    "Which EU law supports agricultural rewilding under GMAL?"
]

selected_question = None
for i, q in enumerate(example_questions):
    if st.button(q, key=f"ex_{i}"):
        selected_question = q

# === Ask User Question ===
query = st.text_input("🔍 Ask a question about EU coastal and restoration laws:")

if selected_question:
    query = selected_question
    st.markdown(f"**Selected Question:** _{query}_")

# === Keyword Filter ===
def is_relevant(q):
    terms = [
        "coastal", "rewilding", "restoration", "habitat", "wetland", "directive", "eu biodiversity",
        "birds", "habitats", "water", "marine", "cap", "climate", "saltmarsh", "kelp", "floodplain",
        "tree canopy", "urban nature", "passive rewilding", "estuaries", "nature-based solutions"
    ]
    return any(term in q.lower() for term in terms)

# === Groq API Chat Function ===
def call_groq_api(question, context=None):
    system_message = (
        "You are an EU legal and ecological assistant helping users understand EU legislation "
        "related to rewilding, restoration, biodiversity, and marine/coastal law under the GMAL framework. "
        "Base answers on the 9 EU directives provided and focus on scenario planning, indicators, "
        "visioning, and legal obligations."
    )
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": question}
        ]
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    else:
        return f"❌ Error {response.status_code}: {response.text}"

# === Handle Response ===
if query:
    if not is_relevant(query):
        st.warning("⚠️ This topic may fall outside the 9 GMAL-related EU directives. Please focus on coastal rewilding, nature, and restoration law.")
    else:
        with st.spinner("💡 Thinking with LLaMA 3..."):
            answer = call_groq_api(query)
            st.subheader("🧠 Answer")
            st.write(answer)
else:
    st.info("💬 Enter a question above or click a suggestion.")
