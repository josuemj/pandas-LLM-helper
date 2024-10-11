import streamlit as st
from core import run_llm

# Set page title and header for better user experience
st.set_page_config(page_title="Langchain bot", page_icon="🤖")
st.title("Langchain assistant")

# Custom CSS for gradienPA  and message styles
st.markdown(
    """
    <style>
    /* Gradient background */
    .main {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 100%);
        color: white;
    }
    
    /* User and AI message styling */
    .user-message {
        background-color: #d1e7dd;
        border-radius: 10px;
        padding: 10px;
        color: #0f5132;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    .ai-message {
        background-color: #f8d7da;
        border-radius: 10px;
        padding: 10px;
        color: #842029;
        font-weight: bold;
        margin-bottom: 10px;
    }

    /* Chatbox customization */
    .stTextInput label {
        color: white;
    }

    /* Change header font size */
    .css-10trblm {
        font-size: 3rem;
        text-align: center;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Initialize or reset session states if not present
if "chat_answer_history" not in st.session_state:
    st.session_state["chat_answer_history"] = []
if "user_prompt_history" not in st.session_state:
    st.session_state["user_prompt_history"] = []
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []
if "response_pending" not in st.session_state:
    st.session_state["response_pending"] = False  # Track whether a response is pending

# Function to format and display the sources of the documents
def create_source_string(source_docs: list) -> str:
    if not source_docs:
        return "No sources available."

    sources_string = "Sources:\n"
    for i, doc in enumerate(source_docs):
        source_info = doc.metadata.get("source", "Unknown source")  # Safely access 'source' from metadata
        sources_string += f"{i+1}. {source_info}\n"
    return sources_string

# Input for the user's prompt
prompt = st.text_input("Ask a question related to langchaing documentation", placeholder="Enter your prompt")

# Handling the response from the LLM
if prompt and not st.session_state["response_pending"]:
    # Set response_pending to True to prevent multiple calls while waiting for the response
    st.session_state["response_pending"] = True

    with st.spinner("Generating response..."):
        generated_response = run_llm(
            query=prompt,
            chat_history=st.session_state["chat_history"]
        )

        # Retrieve the sources from 'generated_response["source"]'
        sources = generated_response.get("source", [])

        # Format the final response with sources
        formatted_response = (
            f"{generated_response['result']}\n\n{create_source_string(sources)}"
        )

        # Update session state for both prompt and response
        st.session_state["user_prompt_history"].append(prompt)
        st.session_state["chat_answer_history"].append(formatted_response)
        st.session_state["chat_history"].append(("human", prompt))
        st.session_state["chat_history"].append(("ai", generated_response["result"]))

        # Reset response_pending once the response is generated
        st.session_state["response_pending"] = False

# Display the chat history
if st.session_state["chat_answer_history"]:
    for i, (user_query, generated_response) in enumerate(zip(st.session_state["user_prompt_history"], st.session_state["chat_answer_history"])):
        # Display user messages with a custom style
        st.markdown(f'<div class="user-message">{user_query}</div>', unsafe_allow_html=True)
        # Display AI responses with a custom style
        st.markdown(f'<div class="ai-message">{generated_response}</div>', unsafe_allow_html=True)
