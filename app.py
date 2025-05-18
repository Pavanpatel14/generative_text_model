import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Streamlit page config (must be first Streamlit command)
st.set_page_config(page_title="GPT-2 Text Generator", page_icon="🤖", layout="centered")

# Load GPT-2 model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

tokenizer, model = load_model()

# Custom CSS for styling with better contrast
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stTextInput>div>div>input {
        font-size: 1.2rem;
        padding: 0.5rem;
    }
    .stButton>button {
        background-color: #4F8BF9;
        color: white;
        font-size: 1.1rem;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
    }
    .generated-text {
        color: #000 !important; /* Black text */
        font-weight: bold;
        font-size: 1.2rem;
        background: #e0e0e0; /* Light gray background */
        border-radius: 8px;
        padding: 1rem;
        margin-top: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🤖 Generative Text Model")
st.write("Enter a prompt !")

prompt = st.text_input("Your prompt", value="Once upon a time")

max_length = st.slider("Max length", 20, 200, 60)
temperature = st.slider("Temperature", 0.5, 1.5, 0.8)

if st.button("Generate"):
    with st.spinner("Generating..."):
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        output = model.generate(
            input_ids,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.95,
            no_repeat_ngram_size=2,
            num_return_sequences=1,
        )
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    st.markdown(
        f'<div class="generated-text"><b>Generated Text:</b><br>{generated_text}</div>',
        unsafe_allow_html=True
    )

st.markdown("---")
st.caption("Made by Patel Pavan❤️")
