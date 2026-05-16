# EdU BOT 🎓

> **An educational AI assistant powered by three fine-tuned transformer models.**  
> Built with ❤️ by **TEAM ELMOLOK** — NLP Final Project

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-14-black?logo=next.js)](https://nextjs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers_4.46-FFD21E)](https://huggingface.co)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📖 Table of Contents

- [About the Project](#-about-the-project)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Models](#-models)
- [Folder Structure](#-folder-structure)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Backend Setup](#step-1--backend-setup)
  - [Frontend Setup](#step-2--frontend-setup)
  - [Production Build](#step-3-optional--production-build)
- [Model Integration](#-model-integration)
- [API Reference](#-api-reference)
- [Screenshots](#-screenshots)
- [Credits](#-credits)

---

## 🧠 About the Project

**EdU BOT** is a full-stack educational AI assistant that combines three state-of-the-art NLP models into one seamless product. It is designed to help students and educators interact with educational content through natural language — asking questions, generating summaries, and creating quizzes automatically.

The backend is built on **FastAPI** for high-performance async inference, and the frontend is a polished **Next.js 14** chat UI with full dark/light mode support and persistent chat history.

---

## ✨ Key Features

| Feature | How to use | What you get |
|---|---|---|
| 🔍 **Question Answering** | Paste a passage, then chat your questions | Extracted answer + confidence score + character offsets in the original passage |
| 📝 **Summarization** | Paste an article, tune `min_length` / `max_length` sliders | Concise summary + word counts, compression %, and inference latency |
| 🧪 **Quiz Generation** | Paste educational text, pick 1–15 questions | Interactive question cards with tap-to-reveal answers and source sentences |
| 💬 **Chat History** | Automatic — nothing to do | Sidebar lists the most recent 50 items with timestamps; one-click clear |
| 🌙 **Dark / Light Mode** | Toggle in the sidebar footer | Persists across reloads via `next-themes` |
| 📌 **Collapsible Sidebar** | Click the chevron icon | Collapse state saved to `localStorage` |

---

## 🏗 System Architecture

```
┌────────────────────────┐         ┌──────────────────────────────┐
│     Next.js 14 UI      │   HTTP  │      FastAPI Inference       │
│  (Tailwind + TS)       │ ──────► │  (BERT · DistilBART · T5)    │
│  - /qa  /summarize     │ ◄────── │  - /api/qa                   │
│  - /quiz  home page    │  JSON   │  - /api/summarize            │
│  - localStorage history│         │  - /api/quiz                 │
└────────────────────────┘         └──────────────────────────────┘
          │                                      │
          └─────────────  http://localhost:3000  ─┘
                                                  ↓
                                    Loads .pth weights from:
                                         models_lg/
                                    ├── qa_model/     (BERT)
                                    ├── sum_model/    (DistilBART)
                                    └── quiz_model/   (T5)
```

The backend loads all three checkpoints **once** at startup using a thread-safe singleton pattern (`ModelManager`), then serves async endpoints for each capability. The frontend proxies all `/api/*` requests to the backend via `next.config.js`, so the browser never crosses origins during development.

---

## 🛠 Tech Stack

### Backend
| Tool | Version | Purpose |
|---|---|---|
| **FastAPI** | 0.115 | Async REST API + auto OpenAPI docs |
| **PyTorch** | 2.x | Model inference engine |
| **Transformers** | 4.46 | HuggingFace model loading & tokenization |
| **Pydantic v2** | latest | Typed request/response validation |
| **pydantic-settings** | latest | `.env` config reader |
| **NLTK** | latest | POS tagging + sentence splitting for quiz pipeline |
| **Uvicorn** | latest | ASGI server |

### Frontend
| Tool | Version | Purpose |
|---|---|---|
| **Next.js** | 14 (App Router) | Full-stack React framework |
| **React** | 18 | UI component library |
| **TypeScript** | strict mode | Type-safe development |
| **Tailwind CSS** | 3.4 | Utility-first styling with custom design tokens |
| **next-themes** | latest | Dark/light mode switching |
| **lucide-react** | latest | Icon library |
| **next/font** | built-in | Self-hosted fonts (Instrument Serif, Plus Jakarta Sans, JetBrains Mono) |

---

## 🤖 Models

| Model | Base Checkpoint | Task | Type |
|---|---|---|---|
| **QA Model** | `deepset/bert-base-uncased-squad2` | Extractive Question Answering | BERT |
| **Summarization Model** | `sshleifer/distilbart-cnn-12-6` | Abstractive Summarization | DistilBART |
| **Quiz Generation Model** | `valhalla/t5-base-qg-hl` | Question Generation with `<hl>` tagging | T5 |

All models:
- Fine-tuned on domain-specific datasets (SQuAD 2.0, CNN/DailyMail)
- Run in `.eval()` mode under `torch.no_grad()` for efficiency
- Automatically use **FP16** on CUDA GPUs for faster inference

---

## 📁 Folder Structure

```
edubot/
├── README.md
├── .gitignore
│
├── backend/                          ← FastAPI inference server
│   ├── app/
│   │   ├── main.py                   ← App entry point + lifespan model loading
│   │   ├── config.py                 ← .env reader (pydantic-settings)
│   │   ├── models/
│   │   │   └── ml_models.py          ← ModelManager singleton (BERT/BART/T5)
│   │   ├── schemas/
│   │   │   └── requests.py           ← Pydantic request/response models
│   │   ├── routers/
│   │   │   ├── qa.py                 ← /api/qa endpoint
│   │   │   ├── summarize.py          ← /api/summarize endpoint
│   │   │   └── quiz.py               ← /api/quiz endpoint
│   │   ├── services/                 ← Actual inference logic
│   │   │   ├── qa_service.py         ← BERT extractive QA + softmax confidence
│   │   │   ├── summarize_service.py  ← DistilBART beam-4 generation
│   │   │   └── quiz_service.py       ← T5 batched <hl>-tag generation
│   │   └── utils/
│   │       └── text_utils.py         ← Sentence split + noun-phrase mining
│   ├── requirements.txt
│   ├── .env.example
│   ├── run.py                        ← `python run.py` shortcut
│   └── README.md
│
├── frontend/                         ← Next.js 14 + Tailwind UI
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── globals.css               ← Design tokens + global utilities
│   │   ├── providers.tsx             ← next-themes wrapper
│   │   ├── page.tsx                  ← Home page (hero + feature cards)
│   │   ├── qa/page.tsx
│   │   ├── summarize/page.tsx
│   │   └── quiz/page.tsx
│   ├── components/
│   │   ├── Sidebar.tsx
│   │   ├── Logo.tsx
│   │   ├── ThemeToggle.tsx
│   │   ├── MessageBubble.tsx
│   │   ├── SummaryCard.tsx
│   │   ├── QuizCard.tsx
│   │   ├── LoadingIndicator.tsx
│   │   ├── CopyButton.tsx
│   │   └── EmptyState.tsx
│   ├── lib/
│   │   ├── api.ts                    ← Typed fetch wrapper over /api/*
│   │   ├── store.ts                  ← localStorage chat history hook
│   │   └── utils.ts
│   ├── tailwind.config.ts
│   ├── tsconfig.json
│   ├── next.config.js                ← /api/* proxy to backend
│   ├── postcss.config.js
│   ├── package.json
│   ├── .env.local.example
│   └── README.md
│
└── models_lg/                        ← ⚠️ You bring this from your notebooks
    ├── qa_model/                     ← Fine-tuned BERT checkpoint
    ├── sum_model/                    ← Fine-tuned DistilBART checkpoint
    └── quiz_model/                   ← Fine-tuned T5 checkpoint
```

---

## 🚀 Getting Started

### Prerequisites

Before you begin, make sure you have the following installed:

- **Python 3.10 or 3.11**
- **Node.js 18+** and **npm**
- A CUDA-capable GPU *(recommended, but not required — the backend falls back to CPU automatically)*
- Your three fine-tuned model checkpoints placed inside `models_lg/` (output of the training notebooks)

---

### Step 1 — Backend Setup

```bash
cd backend

# 1. Create and activate a virtual environment
python -m venv .venv

# Windows:
.venv\Scripts\activate
# macOS / Linux:
# source .venv/bin/activate

# 2. Install all Python dependencies
pip install -r requirements.txt

# 3. Set up the environment config
copy .env.example .env          # Windows
# cp .env.example .env          # macOS / Linux

# 4. Start the inference server
python run.py
```

The server boots on **http://localhost:8000**

Visit **http://localhost:8000/docs** for the interactive Swagger API explorer.

> 💡 **Tip:** If your `models_lg/` folder lives somewhere else (e.g., `D:\models_lg`), open `backend/.env` and set the absolute paths — there are comments inside the file to guide you.

---

### Step 2 — Frontend Setup

Open a **second terminal** and run:

```bash
cd frontend

# 1. Install Node.js dependencies
npm install

# 2. Set up environment config
copy .env.local.example .env.local    # Windows
# cp .env.local.example .env.local   # macOS / Linux

# 3. Start the dev server
npm run dev
```

Open your browser and navigate to **http://localhost:3000**

That's it — you can now ask questions, summarize articles, and generate quizzes! 🎉

---

### Step 3 (Optional) — Production Build

```bash
# ── Frontend ──────────────────────────────────────────
cd frontend
npm run build
npm start                  # Serves on :3000

# ── Backend ───────────────────────────────────────────
cd backend
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> ⚠️ **Important:** Always use `--workers 1` for the backend in production.  
> Model weights are loaded into process memory — replicating them per worker multiplies RAM/VRAM usage.  
> For higher throughput, scale **horizontally** with a process manager, or use `model.share_memory()` with multiple workers.

---

## 🔌 Model Integration

The training notebooks save each fine-tuned model using HuggingFace's standard `save_pretrained()` method. Place the output folders side-by-side under `models_lg/` and the backend will find them automatically via the `.env` paths.

```python
# backend/app/models/ml_models.py — ModelManager singleton (abbreviated)
self.qa_tokenizer, self.qa_model = self._load_one(
    "QA", settings.QA_MODEL_PATH, AutoModelForQuestionAnswering,
)
self.sum_tokenizer, self.sum_model = self._load_one(
    "Summarizer", settings.SUM_MODEL_PATH, AutoModelForSeq2SeqLM,
)
self.quiz_tokenizer, self.quiz_model = self._load_one(
    "Quiz", settings.QUIZ_MODEL_PATH, AutoModelForSeq2SeqLM,
)
```

### How each service works

#### 🔍 `qa_service.py` — Extractive Question Answering
- Tokenizes input with `max_length=384`, `truncation='only_second'`
- Mirrors the exact tokenization settings from the training notebook
- Adds a **softmax-based confidence score** on top of BERT logits
- Recovers **character offsets** from `return_offsets_mapping` so the UI can highlight the answer span in the original passage

#### 📝 `summarize_service.py` — Abstractive Summarization
- Generates using the notebook's original parameters: `num_beams=4`, `early_stopping=True`
- Adds `length_penalty` and `no_repeat_ngram_size=3` to keep summaries concise and non-repetitive
- Returns word counts, compression percentage, and inference latency

#### 🧪 `quiz_service.py` — Question Generation (most complex)
The T5 checkpoint expects inputs with `<hl> ... <hl>` tags surrounding the answer span. The service handles this through a multi-step pipeline:

1. **Sentence splitting** — uses NLTK to break the passage into individual sentences
2. **POS tagging** — identifies noun-phrases and numbers as candidate answers
3. **`<hl>` wrapping** — wraps each candidate using the exact `make_qg_input` rule from the training notebook
4. **Batched inference** — runs T5 in **batches of 8** with `num_beams=4`
5. **De-duplication** — filters repeated questions and stops at the requested count

---

## 📡 API Reference

### POST `/api/qa` — Question Answering
```json
Request:
{
  "question": "When was the Eiffel Tower built?",
  "context":  "The Eiffel Tower is a wrought-iron lattice tower..."
}

Response:
{
  "answer": "1887",
  "confidence": 0.94,
  "start": 42,
  "end": 46
}
```

### POST `/api/summarize` — Summarization
```json
Request:
{
  "text": "Long article text here...",
  "max_length": 150,
  "min_length": 40
}

Response:
{
  "summary": "Concise summary here...",
  "original_word_count": 520,
  "summary_word_count": 87,
  "compression_percent": 83.3,
  "latency_ms": 1240
}
```

### POST `/api/quiz` — Quiz Generation
```json
Request:
{
  "text": "Educational passage here...",
  "num_questions": 5
}

Response:
{
  "questions": [
    {
      "question": "What is the capital of France?",
      "answer": "Paris",
      "source_sentence": "Paris is the capital of France..."
    }
  ]
}
```

### GET `/health` — Health Check
```json
{
  "status": "ok",
  "version": "1.0.0",
  "team": "TEAM ELMOLOK",
  "models": {
    "device": "cuda",
    "fp16": true
  }
}
```

### GET `/docs` — Interactive Swagger UI
Visit `http://localhost:8000/docs` for the full interactive API explorer.

---

## 🖼 Screenshots

> Add screenshots here after deployment. Suggested shots:
> - Home page hero section
> - QA chat interface with a question answered
> - Summarization page with sliders
> - Quiz cards with tap-to-reveal answers
> - Dark mode vs. light mode comparison

---

## 🙏 Credits

Built for the **EDU_BOT NLP Final Project** by **TEAM ELMOLOK**.

### Datasets
| Dataset | Authors | Used For |
|---|---|---|
| **SQuAD 2.0** | Rajpurkar et al. | QA model fine-tuning |
| **CNN / DailyMail** | Hermann et al. | Summarization model fine-tuning |

### Base Checkpoints (HuggingFace Hub)
| Checkpoint | Model | Task |
|---|---|---|
| `deepset/bert-base-uncased-squad2` | BERT-base | Question Answering |
| `sshleifer/distilbart-cnn-12-6` | DistilBART | Summarization |
| `valhalla/t5-base-qg-hl` | T5-base | Quiz / Question Generation |

---

<div align="center">

Made with ❤️ by **TEAM ELMOLOK**

</div>
