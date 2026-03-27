import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables for local testing
load_dotenv()

# --- 1. SYSTEM PROMPT CONFIGURATION ---
# This defines the "Nature Educator" persona and curriculum constraints.
SYSTEM_MESSAGE = """
You are Nature Educator, a specialized AI learning companion for the Kenneth Anderson Nature Society (KANS) in Krishnagiri. 

1. Core Identity: Curious, knowledgeable, and encouraging guide.
2. Knowledge: Strictly limited to KANS curriculum (Biodiversity, Water, Waste) and Krishnagiri local context.
3. Personas: 
   - Classes 7-9: Simple language, step-by-step guidance.
   - Class 11: Sophisticated language, data analysis support.
4. Safety: No PII, no personal/emotional talk, strictly educational.
5. Gamification: Use "Challenges" and "Badges" to mark progress toward "KANS Climate Champion."
"""

# --- 2. APP UI SETUP ---
st.set_page_config(page_title="KANS Nature Education", page_icon="🌱")

# Display Logos (Handles missing files gracefully)
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
try:
    with col2: st.image("assets/kans_logo.png", width=120)
    with col4: st.image("assets/cnws_logo.png", width=120)
except Exception:
    pass

st.title("KANS Nature Education Program")
st.caption("Empowering Krishnagiri's students to become Climate Champions.")
st.markdown("---")

# --- 3. BOT LOGIC ---
def get_response(user_query, chat_history):
    # Retrieve API Key from Secrets (Cloud) or Environment (Local)
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("API Key not found. Please add GOOGLE_API_KEY to Streamlit Secrets.")
        st.stop()

    # The prompt now ONLY handles the conversation flow. 
    # The System Instructions are handled by the LLM object directly.
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_question}"),
    ])

    # OPTION 1: Passing instructions via 'system_instruction'
    # This is the most stable way to use Gemini with LangChain.
    llm = ChatGoogleGenerativeAI(
        model="gemini-3-flash-preview",
        system_instruction=SYSTEM_MESSAGE, 
        google_api_key=api_key,
        temperature=0.3, # Low temperature for factual consistency
        max_retries=2
    )

    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

# --- 4. SESSION STATE & HISTORY ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi there! I'm your Nature Educator. I'm here to help you on your journey to becoming a KANS Climate Champion. What grade and school are you from?"),
    ]

# Render existing chat
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# --- 5. INTERACTION ---
user_query = st.chat_input("Type your question about nature here...")

if user_query:
    # 1. Update UI with Human Message
    with st.chat_message("Human"):
        st.markdown(user_query)

    # 2. Generate and stream AI Response
    with st.chat_message("AI"):
        # We pass history EXCEPT the message we just added (to avoid duplication in the chain)
        full_response = st.write_stream(get_response(user_query, st.session_state.chat_history))

    # 3. Save both to session state (ensuring they are plain strings/objects)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=str(full_response)))
