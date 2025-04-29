import streamlit as st
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

st.set_page_config(page_title="GenAI Math App", layout="centered")
st.title("🧠 Generative AI for Creative Math Tasks")
st.sidebar.title("Choose a Task")

# Sidebar selection
task = st.sidebar.radio("Select a module:", ["Math Riddle Generator", "Math Meme Repair", "Emoji Math Solver"])

# --- Common Function to load a model ---
@st.cache_resource(show_spinner=False)
def load_model(model_path, use_4bit=False):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=use_4bit,
        load_in_8bit=not use_4bit,
        llm_int8_enable_fp32_cpu_offload=True
    )
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=bnb_config).to("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer

# --- Q1: Math Riddle Generator ---
if task == "Math Riddle Generator":
    st.subheader("🧠 Math Riddle Answer Generator")
    question = st.text_input("Ask a math riddle:")
    
    if "model_q1" not in st.session_state:
        with st.spinner("Loading Riddle Model..."):
            st.session_state.model_q1, st.session_state.tokenizer_q1 = load_model("fine-tuned-QA-tinyllama-1.1B")

    if st.button("✨ Generate Answer"):
        if question.strip():
            prompt = f"Question: {question} Answer:"
            inputs = st.session_state.tokenizer_q1(prompt, return_tensors="pt").to(st.session_state.model_q1.device)
            with torch.no_grad():
                output = st.session_state.model_q1.generate(**inputs, max_length=50, temperature=0.7, top_k=50, top_p=0.9)
            response = st.session_state.tokenizer_q1.decode(output[0], skip_special_tokens=True)
            st.markdown("**Answer:** " + response.split("Answer:")[-1].strip())
        else:
            st.warning("Please enter a riddle to generate an answer.")

# --- Q2: Math Meme Repair ---
elif task == "Math Meme Repair":
    st.subheader("🧩 Math Meme Explanation Fixer")
    meme = st.text_area("Paste your broken math meme (text):", height=150)

    if "model_q2" not in st.session_state:
        with st.spinner("Loading Meme Repair Model..."):
            st.session_state.model_q2, st.session_state.tokenizer_q2 = load_model("deepseek")

    if st.button("🛠️ Fix Meme"):
        if meme.strip():
            prompt = f"""Below is a math meme with an incorrect solution. Your task is to identify the error and provide a correct explanation.

### Incorrect Meme:
{meme}

### Identified Error:"""
            inputs = st.session_state.tokenizer_q2(prompt, return_tensors="pt").to(st.session_state.model_q2.device)
            with torch.no_grad():
                output = st.session_state.model_q2.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=150,
                    do_sample=True,
                    temperature=0.3
                )
            response = st.session_state.tokenizer_q2.decode(output[0], skip_special_tokens=True)
            st.markdown("**Fixed Explanation:**")
            st.write(response.split("### Fixed Explanation:")[-1].strip())
        else:
            st.warning("Please paste a meme description to fix.")

# --- Q3: Emoji Math Solver ---
elif task == "Emoji Math Solver":
    st.subheader("🔢 Solve Emoji Math Problems")
    emoji_question = st.text_area("Enter an emoji-style math question:", height=150)

    if "model_q3" not in st.session_state:
        with st.spinner("Loading Emoji Math Model..."):
            st.session_state.model_q3, st.session_state.tokenizer_q3 = load_model("DeepSeek-R1-Medical-COT")

    if st.button("🧮 Solve"):
        if emoji_question.strip():
            prompt = f"""Below is an instruction that describes a task, paired with an input that provides further context.
Write a response that appropriately completes the request.
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Instruction:
You are a math expert with advanced knowledge in mathematical reasoning, problem-solving, and theoretical concepts.
Please answer the following math question. The response should include an answer and reasoning, e.g., '🍎 = 4'

### Question:
{emoji_question}

### Response:
<think>"""

            inputs = st.session_state.tokenizer_q3(prompt, return_tensors="pt").to(st.session_state.model_q3.device)
            with torch.no_grad():
                output = st.session_state.model_q3.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=150,
                    temperature=0.5
                )
            response = st.session_state.tokenizer_q3.decode(output[0], skip_special_tokens=True)
            st.markdown("**Solution:**")
            st.write(response.split("### Response:")[-1].strip())
        else:
            st.warning("Please enter an emoji math problem.")
