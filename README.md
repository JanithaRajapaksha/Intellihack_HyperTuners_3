# 🚀 Fine-Tuned Qwen 2.5 3B Deployment Guide

## **1️⃣ Model Fine-Tuning in Google Colab**
1. Open the **Colab Notebook** and run all cells to:
   - Fine-tune the **Qwen 2.5 3B** model.
   - Convert the model to **4-bit GGUF format**.
   - Download the final **GGUF model file**.

2. Ensure the downloaded **GGUF file** is saved locally for deployment.

---

## **2️⃣ Install Ollama & Prepare the Model**
1. **Install Ollama** (if not already installed):
   ```bash
   curl -fsSL https://ollama.ai/install.sh | sh
   ```

2. **Download & Place the GGUF Model File**: Move the fine-tuned GGUF model (e.g., `qwen3b_finetuned.gguf`) to:
   ```bash
   ~/.ollama/models/
   ```

3. **Create an Ollama Model File (Modelfile)**:
   ```plaintext
   FROM /path/to/qwen3b_finetuned.gguf
   ```
   Replace `/path/to/qwen3b_finetuned.gguf` with the actual file path.

4. **Register the Model in Ollama**:
   ```bash
   ollama create qwen-finetuned -f Modelfile
   ```

---

## **3️⃣ Install Python Dependencies**
Run the following command to install all necessary dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## **4️⃣ Running the Local Deployment**
Once everything is set up, start the local AI system:
   ```bash
   python app.py
   ```
This will:
   - Load the fine-tuned **Qwen 2.5 3B GGUF model**.
   - Use **Gemma-2B** for reasoning.
   - Retrieve context using **ChromaDB**.
   - Provide a **Gradio UI** for user queries.

---

## **5️⃣ Testing the Model**
You can send a test query using Python:
   ```python
   from primary_agent import get_primary_agent
   
   agent = get_primary_agent()
   response = agent.run("Explain Reasoning-Oriented Reinforcement Learning.")
   
   print("Response:", response)
   ```

---

## **6️⃣ Troubleshooting**
- **Model Not Found?** Ensure the GGUF model is correctly placed in `~/.ollama/models/`.
- **Ollama Fails to Load?** Try restarting the Ollama service:
   ```bash
   ollama serve
   ```
- **Out of Memory Issues?** Reduce `n_ctx` when loading the GGUF model.