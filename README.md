# AutoStream Conversational AI Agent

A production-grade, multi-turn conversational AI agent that converts product inquiries into qualified leads using **RAG (Retrieval-Augmented Generation)**, **LangGraph**, and **Google Gemini**.

---

## ✨ Features

- 🧠 **Continuous Intent Classification** — Re-evaluated on every turn; gracefully re-routes mid-conversation
- 📚 **RAG-Powered Q&A** — FAISS vector search over a local knowledge base with anti-hallucination guardrails
- 🎯 **Structured Lead Capture** — Pydantic schema extraction for reliable, validated field collection
- 🔒 **Tool Guard Logic** — Lead capture tool fires *only* after all required fields are collected
- 💾 **Multi-Turn Memory** — LangGraph `MemorySaver` persists full conversation state across turns
- 🆓 **Fully Local Embeddings** — `all-MiniLM-L6-v2` runs on-device; no embedding API costs
- 🖥️ **Rich Terminal UI** — Beautiful CLI with live state diagnostics powered by `rich`

---

## 🗂️ Project Structure

```
autostream-agent/
├── agent/
│   ├── __init__.py
│   ├── graph.py          # LangGraph StateGraph + MemorySaver compilation
│   ├── nodes.py          # 5 node functions: intent, RAG, Pydantic, capture, generate
│   └── state.py          # AgentState TypedDict
├── knowledge_base/
│   └── autostream_kb.md  # Product knowledge base (pricing, features, policies)
├── rag/
│   ├── __init__.py
│   ├── embedder.py       # FAISS index builder using MiniLM-L6-v2
│   └── retriever.py      # Semantic search with relevance threshold
├── tools/
│   ├── __init__.py
│   └── lead_capture.py   # mock_lead_capture() tool
├── main.py               # CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

> **Note:** `faiss_index/` is gitignored — it is built automatically on first run.

---

## 🚀 Local Setup

### Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10 or higher |
| pip | latest (bundled with Python) |
| Google Gemini API key | Free — [get one here](https://aistudio.google.com/app/apikey) |

### Step-by-Step Installation

**1. Clone the repository**
```bash
git clone https://github.com/your-username/autostream-agent.git
cd autostream-agent
```

**2. Create and activate a virtual environment**
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate.bat

# Windows (PowerShell)
python -m venv venv
venv\Scripts\Activate.ps1
```

**3. Install all dependencies**
```bash
pip install -r requirements.txt
```

This will install:
- `langgraph` + `langchain` — agent framework
- `langchain-google-genai` — Gemini LLM integration
- `faiss-cpu` — local vector store
- `sentence-transformers` — local embeddings (`all-MiniLM-L6-v2`)
- `rich` — terminal UI
- `pydantic` — structured output validation
- `python-dotenv` — `.env` file loading

> ⏱️ **First install may take 2–3 minutes** — `sentence-transformers` downloads the embedding model (~90 MB) on first use.

**4. Configure your API key**
```bash
cp .env.example .env
```

Open `.env` and replace the placeholder with your actual Gemini API key:
```
GOOGLE_API_KEY=your_actual_key_here
```

**5. Run the agent**
```bash
python main.py
```

On **first run**, the FAISS index is automatically built from `knowledge_base/autostream_kb.md`. Subsequent runs load the cached index instantly.

---

## 🎬 Example Conversation

```
You: Hi, tell me about your pricing.

🤖 Alex: AutoStream has two plans! The Basic plan is $29/month (10 videos/month,
   720p). The Pro plan is $79/month with unlimited videos, 4K, and AI captions.
   Both come with a 30-day free trial!

You: I want to try the Pro plan for my YouTube channel.

🤖 Alex: Fantastic! I'd love to get you set up. What's your full name?

You: Jane Doe

🤖 Alex: Great, Jane! What email should we use to contact you?

You: jane@example.com

🤖 Alex: Perfect. And what platform are you primarily creating content for?

You: YouTube

🤖 Alex: Welcome to AutoStream, Jane! 🎉 You're all set.
   Name: Jane Doe | Email: jane@example.com | Platform: YouTube.
   Our team will reach out within 24 hours!

   [Lead Status: ✅ Captured — CRM updated]
```

**Special CLI commands:**
- `reset` — Start a fresh conversation thread
- `quit` / `exit` / `bye` — End the session

---

## 🏗️ Architecture

```
START → classify_intent (every turn)
  ├─ greeting      → generate_response
  ├─ inquiry       → retrieve_knowledge → generate_response
  └─ high_intent   → collect_lead_info → [capture_lead →] generate_response
```

The agent uses **LangGraph** for stateful, conditional routing across conversation turns. The RAG pipeline embeds documents locally using `all-MiniLM-L6-v2` stored in a FAISS `IndexFlatIP` (cosine similarity). A **relevance threshold** (≥ 0.25) prevents low-quality context from reaching the LLM, and the system prompt enforces strict grounding — the agent never answers from outside its knowledge base.

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph 0.2+ |
| LLM | Google Gemini (via `langchain-google-genai`) |
| Local Embeddings | `all-MiniLM-L6-v2` (sentence-transformers) |
| Vector Store | FAISS (`IndexFlatIP`) — fully local |
| Structured Output | Pydantic v2 |
| Terminal UI | Rich |

---

## 📄 License

MIT
