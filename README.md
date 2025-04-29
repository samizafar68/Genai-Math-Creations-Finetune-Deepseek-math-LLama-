# 🚀 GenAI Math App — Generative AI for Creative Math Challenges

A Streamlit web application that demonstrates the power of generative AI in solving and creating math problems. Built end-to-end with PyTorch, Hugging Face Transformers, PEFT/LoRA, and Unsloth quantization for lightweight, production-ready inference.

---

## 🌟 Features & Modules

1. 🎲 **Math Riddle Generator**  
   - Generates and solves logic-based math riddles (e.g. “What number becomes zero when you subtract 15 from half of it?”).  
   - Powered by a TinyLlama (1.1B) model fine-tuned via LoRA.

2. 🧩 **Math Meme Repair**  
   - “Fixes” viral math memes (e.g. “8 ÷ 2(2+2) = 1?”) by explaining the mistake and providing the correct solution.  
   - Uses a DeepSeek-based model fine-tuned on meme data.

3. 🔢 **Emoji Math Solver**  
   - Interprets and solves math problems written in emoji (e.g. 🍎 + 🍎 + 🍎 = 12 → 🍎 = 4).  
   - Employs a DeepSeek model with custom prompt-engineering and 4-bit quantization.

---

## 🛠️ Tech Stack

- **PyTorch & Transformers** — model implementation and inference  
- **PEFT & LoRA** — efficient fine-tuning of large language models  
- **Unsloth** — easy low-bit (4/8-bit) quantization and loading  
- **Streamlit** — rapid web UI for live interaction  
- **Hugging Face Datasets** — data handling and processing  

---
