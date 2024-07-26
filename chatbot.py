from dotenv import load_dotenv
import streamlit as st
import os
import cohere
from trulens_eval import TruChain, Feedback, Huggingface, Tru
import trulens_eval


# Load environment variables from .env file
load_dotenv()

cohere_api_key = os.getenv("COHERE_API_KEY")
huggingface_api_key = os.getenv("HUGGINGFACE_API_KEY")

# Initialize the Cohere client
cohere_client = cohere.Client(cohere_api_key)

hugs = Huggingface(api_key=huggingface_api_key)
tru = Tru()

# Define a function to generate text using Cohere with increased max_tokens and other parameters
def generate_text_with_cohere(prompt, model='command', max_tokens=4000, temperature=0.7):
    try:
        response = cohere_client.generate(
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.generations[0].text.strip()
    except cohere.NotFoundError as e:
        print(f"Model not found: {e}")
        return "Error: Model not found"

# Define custom relevance and moderation functions
def custom_relevance(input, output):
    return 1.0

def custom_moderation_hate(output):
    return 0.0

def custom_moderation_violence(output):
    return 0.0

def custom_moderation_selfharm(output):
    return 0.0

def custom_moderation_maliciousness(output):
    return 0.0

# Define feedbacks
f_relevance = Feedback(custom_relevance).on_input_output()
f_hate = Feedback(custom_moderation_hate).on_output()
f_violent = Feedback(custom_moderation_violence, higher_is_better=False).on_output()
f_selfharm = Feedback(custom_moderation_selfharm, higher_is_better=False).on_output()
f_maliciousness = Feedback(custom_moderation_maliciousness, higher_is_better=False).on_output()

# Define TruLens evaluation chain
chain_recorder = TruChain(
    generate_text_with_cohere, app_id="contextual-chatbot", feedbacks=[f_relevance, f_hate, f_violent, f_selfharm, f_maliciousness]
)

# Streamlit frontend
st.title("Contextual Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Record with TruLens
        with chain_recorder as recording:
            full_response = generate_text_with_cohere(prompt, model='command')  # Adjust model as needed
        message_placeholder = st.empty()
        message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append(
        {"role": "assistant", "content": full_response}
    )

# Run TruLens dashboard
tru.run_dashboard()
