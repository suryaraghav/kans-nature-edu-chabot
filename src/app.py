import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
import os

# Load variables from .env if running locally
load_dotenv()

# --- 1. SYSTEM PROMPT CONFIGURATION ---
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

# --- 2. APP CONFIG & UI ---
st.set_page_config(page_title="KANS Nature Education", page_icon="🌱")

# Logos (Ensure these paths match your 'assets' folder in GitHub)
col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 1])
with col2:
    try:
        st.image("assets/kans_logo.png", width=120)
    except:
        pass
with col4:
    try:
        st.image("assets/cnws_logo.png", width=120)
    except:
        pass

st.title("KANS Nature Education Program")
st.markdown("---")

# --- 3. CORE CHAT LOGIC ---
def get_response(user_query, chat_history):
    # Retrieve API Key from Streamlit Secrets or Environment
    api_key = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error("Missing Google API Key. Please add it to Streamlit Secrets.")
        st.stop()

    # Define the prompt structure
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_MESSAGE),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{user_question}"),
    ])

    # Initialize LLM (Using a verified stable model string)
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=api_key,
        temperature=0.3,
        max_retries=2
    )

    chain = prompt | llm | StrOutputParser()
    
    return chain.stream({
        "chat_history": chat_history,
        "user_question": user_query,
    })

# --- 4. SESSION STATE MANAGEMENT ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        AIMessage(content="Hi there! I'm your Nature Educator. I'm here to help you become a KANS Climate Champion. What grade and school are you from?"),
    ]

# Display conversation history
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.markdown(message.content)

# --- 5. USER INPUT HANDLING ---
user_query = st.chat_input("Ask about Biodiversity, Water, or Waste...")

if user_query:
    # Add user message to history
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message("Human"):
        st.markdown(user_query)

    with st.chat_message("AI"):
        # st.write_stream consumes the generator and returns the final string
        # This prevents the 'AIMessage content expected str' ValidationError
        full_response = st.write_stream(get_response(user_query, st.session_state.chat_history[:-1]))

    # Append the resolved string to history
    st.session_state.chat_history.append(AIMessage(content=str(full_response)))
