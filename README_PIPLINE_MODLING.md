# 🧠 NLP Pipeline: QA · Summarization · Quiz Generation

> **Distributed training across 3 machines** — Each notebook ran independently on a separate GPU-equipped machine. Models are saved to `./models_lg/` and can be loaded together for a unified inference pipeline.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Distributed Training Setup](#distributed-training-setup)
- [Notebook 1 — BERT vs RNN vs LSTM (Bonus QA)](#notebook-1--bert-vs-rnn-vs-lstm-bonus-qa)
- [Notebook 2 — BERT + DistilBART Pipeline](#notebook-2--bert--distilbart-pipeline)
- [Notebook 3 — T5 Quiz Generation](#notebook-3--t5-quiz-generation)
- [Datasets](#datasets)
- [Models Summary](#models-summary)
- [Evaluation Metrics](#evaluation-metrics)
- [How to Run](#how-to-run)
- [Dependencies](#dependencies)

---

## Project Overview

This project builds a full NLP pipeline covering **three core tasks**:

| Task | Model | Dataset |
|------|-------|---------|
| Extractive Question Answering | BERT (`deepset/bert-base-uncased-squad2`) | SQuAD |
| Abstractive Summarization | DistilBART (`sshleifer/distilbart-cnn-12-6`) | CNN/DailyMail |
| Quiz / Question Generation | T5 (`valhalla/t5-base-qg-hl`) | SQuAD |

A **bonus comparative study** was also conducted pitting BERT against custom RNN and LSTM models on the same QA task, providing a clear benchmark of classical sequence models vs. Transformers.

---

## Repository Structure

```
project/
│
├── bonus_rnn_lstm_qa.ipynb          # Machine 1 — BERT vs RNN vs LSTM (Span Extraction)
├── pipeline_BERT_DistilBART.ipynb   # Machine 2 — BERT (QA) + DistilBART (Summarization)
├── pipeline_T5_only.ipynb           # Machine 3 — T5 (Quiz Generation)
│
├── models_lg/
│   ├── qa_model/                    # Fine-tuned BERT for QA
│   ├── sum_model/                   # Fine-tuned DistilBART for Summarization
│   └── quiz_model/                  # Fine-tuned T5 for Quiz Generation
│
└── qa_span_extraction_comparison.png  # Bonus comparison chart
```

---

## Distributed Training Setup

Training was split across **3 separate machines** to handle the computational load:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Machine 1                                                           │
│  bonus_rnn_lstm_qa.ipynb                                            │
│  → Trains RNN & LSTM from scratch                                   │
│  → Evaluates vs pre-trained BERT (no fine-tuning needed)            │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  Machine 2                                                           │
│  pipeline_BERT_DistilBART.ipynb                                     │
│  → Fine-tunes BERT on SQuAD (QA task)                              │
│  → Fine-tunes DistilBART on CNN/DailyMail (Summarization)          │
│  → Saves both models to ./models_lg/                                │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│  Machine 3                                                           │
│  pipeline_T5_only.ipynb                                             │
│  → Fine-tunes T5 on SQuAD (Quiz Generation task)                   │
│  → Saves model to ./models_lg/quiz_model/                           │
└─────────────────────────────────────────────────────────────────────┘
```

Each machine produces independently saved model checkpoints. Load them together to run the full pipeline.

---

## Notebook 1 — BERT vs RNN vs LSTM (Bonus QA)

**File:** `bonus_rnn_lstm_qa.ipynb`  
**Machine:** 1  
**Task:** Extractive QA — predict answer **start & end token positions** (0–384)

### What it does

Compares three fundamentally different architectures on the same SQuAD task:

| Model | Type | Architecture |
|-------|------|-------------|
| BERT | Transformer (Pre-trained) | `deepset/bert-base-uncased-squad2` — zero fine-tuning |
| LSTM | Recurrent (Trained from scratch) | BiLSTM → Dense → Start/End Softmax |
| RNN | Recurrent (Trained from scratch) | BiRNN → Dense → Start/End Softmax |

### Data Pipeline

```
SQuAD (full train split)
  → Clean text (strip + normalize whitespace)
  → Remove outliers (IQR method on question/context/answer lengths)
  → Keep top-30 most frequent answers only
  → Sample 5,000 examples (random_state=42)
  → 80/20 Train/Test split
  → Evaluate on same 100-sample subset for fair comparison
```

### Model Architecture (RNN & LSTM)

```
Input (token IDs, max_len=128)
  → Embedding (vocab_size × 128)
  → Bidirectional RNN/LSTM (64 units, return_sequences=True)
  → Dropout (0.3)
  → Dense (64, relu)
  → Two parallel heads:
       start_logits → Dense(1) → Squeeze → Softmax  →  start position
       end_logits   → Dense(1) → Squeeze → Softmax  →  end position
```

- **Optimizer:** Adam  
- **Loss:** Sparse Categorical Crossentropy (on both heads)  
- **Epochs:** 5 | **Batch Size:** 32  

### Expected Results Ranking

```
🥇 BERT   — Best (pre-trained on massive corpora, self-attention)
🥈 LSTM   — Better memory than RNN (handles long-range dependencies)
🥉 RNN    — Baseline (suffers from vanishing gradients)
```

### Output

- Console comparison table (EM%, F1%)
- `qa_span_extraction_comparison.png` — bar charts + training curves

---

## Notebook 2 — BERT + DistilBART Pipeline

**File:** `pipeline_BERT_DistilBART.ipynb`  
**Machine:** 2  
**Tasks:** QA (BERT) + Summarization (DistilBART)

> T5 (Quiz Generation) is declared and loaded here but **trained in Notebook 3** to distribute the workload.

### EDA Performed

- Word length distributions (histplot) for questions, contexts, answers, articles, summaries
- Box plots for outlier detection
- Compression ratio analysis (summary_len / article_len) — before & after filtering
- Question type distribution (`What`, `Who`, `When`, `Where`, `How`, `Why`, `Which`)
- Word clouds for SQuAD questions and CNN/DM summaries

### Preprocessing

```
1. strip() + normalize whitespace (re.sub \s+)
2. Drop null rows
3. IQR-based outlier removal on all length columns
4. CNN/DM: keep only rows with compression_ratio < 1
```

### Input Formatting

**QA (BERT):**
```
question [SEP] context  →  start_position, end_position
```

**Quiz Generation (T5 format — prepared here, trained in NB3):**
```
generate question: {before}<hl> {answer} <hl>{after}  →  question text
```

### Data Splits

| Dataset | Train | Test |
|---------|-------|------|
| SQuAD (QA + Quiz) | 13,000 | 3,000 |
| CNN/DailyMail (Summarization) | 8,000 | 1,800 |

### Training — BERT (QA)

```python
TrainingArguments(
    num_train_epochs            = 5,
    learning_rate               = 3e-5,
    per_device_train_batch_size = 6,
    per_device_eval_batch_size  = 8,
    eval_strategy               = 'epoch',
    save_strategy               = 'epoch',
    load_best_model_at_end      = True,
    fp16                        = True,       # Mixed precision
    warmup_steps                = 100,
    weight_decay                = 0.01,
    early_stopping_patience     = 2
)
```

### Training — DistilBART (Summarization)

```python
Seq2SeqTrainingArguments(
    num_train_epochs            = 3,
    learning_rate               = 3e-5,
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 2,          # Effective batch = 4
    eval_strategy               = 'epoch',
    predict_with_generate       = True,
    fp16                        = True,
    warmup_steps                = 200,
    weight_decay                = 0.01,
    early_stopping_patience     = 2
)
```

### Custom Dataset Classes

- **`QADataset`** — Handles offset mapping to align character-level answer positions with token-level start/end positions. Automatically skips samples where the answer cannot be located in context.
- **`Seq2SeqDataset`** — General-purpose seq2seq tokenizer wrapper with label padding masked to `-100` (ignored by cross-entropy loss).

### Evaluation

| Model | Metric | Notes |
|-------|--------|-------|
| BERT | Exact Match (EM%), F1% | Evaluated on 100 test samples via `evaluate.load("squad")` |
| DistilBART | ROUGE-1/2/L | Evaluated on 40 test samples via `evaluate.load("rouge")` |

### Saved Models

```
./models_lg/qa_model/       ← BERT weights + tokenizer
./models_lg/sum_model/      ← DistilBART weights + tokenizer
./models_lg/quiz_model/     ← T5 weights + tokenizer (saved here, trained in NB3)
```

Training curves (loss per epoch) are plotted for all 3 models from `trainer_state.json`.

---

## Notebook 3 — T5 Quiz Generation

**File:** `pipeline_T5_only.ipynb`  
**Machine:** 3  
**Task:** Question Generation from highlighted answer spans

### Why a separate notebook?

DistilBART training on Machine 2 consumes significant VRAM. T5 training was offloaded to Machine 3 to run in parallel, then its saved weights are merged back into `./models_lg/quiz_model/`.

### Model

**`valhalla/t5-base-qg-hl`** — A T5-base model pre-trained specifically for question generation using highlight markers (`<hl>`).

### Input Format

The answer span is highlighted inside the context:

```
generate question: The Normans were a people {before}<hl> Viking <hl>{after}

→ Output: "What were the Normans descended from?"
```

### Data

```
SQuAD (full train split)
  → Same preprocessing as Notebook 2
  → Train: 13,000 | Test: 3,000
  → Max input length: 512 tokens
  → Max target length: 64 tokens
```

### Training — T5 (Quiz Generation)

```python
Seq2SeqTrainingArguments(
    num_train_epochs            = 3,
    learning_rate               = 3e-5,
    per_device_train_batch_size = 4,
    per_device_eval_batch_size  = 8,
    eval_strategy               = 'epoch',
    predict_with_generate       = True,
    fp16                        = True,
    warmup_steps                = 200,
    weight_decay                = 0.01,
    early_stopping_patience     = 2
)
```

### Evaluation

```
ROUGE-1   →  Unigram overlap between generated and reference question
ROUGE-2   →  Bigram overlap
ROUGE-L   →  Longest common subsequence
BLEU-4    →  4-gram precision with smoothing (method1)
```

Evaluated using `rouge_scorer` and `nltk.translate.bleu_score` on the test set.

### Saved Model

```
./models_lg/quiz_model/     ← T5 weights + tokenizer
```

---

## Datasets

| Dataset | Task | Size (used) | Source |
|---------|------|-------------|--------|
| **SQuAD** | QA + Quiz Generation | ~87,000 train (filtered to 13K) | `datasets` library |
| **CNN/DailyMail 3.0.0** | Summarization | 20,000 (filtered to 8K) | `datasets` library |

---

## Models Summary

| Model | Base Checkpoint | Task | Training |
|-------|----------------|------|----------|
| BERT | `deepset/bert-base-uncased-squad2` | Extractive QA | Fine-tuned (NB2) |
| DistilBART | `sshleifer/distilbart-cnn-12-6` | Summarization | Fine-tuned (NB2) |
| T5 | `valhalla/t5-base-qg-hl` | Quiz Generation | Fine-tuned (NB3) |
| BiLSTM | — (from scratch) | Extractive QA | Trained (NB1) |
| BiRNN | — (from scratch) | Extractive QA | Trained (NB1) |

---

## Evaluation Metrics

### Question Answering
- **Exact Match (EM):** % of predictions that exactly match the ground truth answer string
- **F1 Score:** Token-level F1 between predicted and reference answer

### Summarization
- **ROUGE-1 / ROUGE-2 / ROUGE-L:** Recall-Oriented Understudy for Gisting Evaluation — measures n-gram and sequence overlap between generated and reference summaries

### Quiz Generation
- **ROUGE-1 / ROUGE-2 / ROUGE-L:** Same as above, applied to generated questions
- **BLEU-4:** 4-gram precision score with smoothing — measures how close the generated question is to the reference

---

## How to Run

### Step 1 — Install dependencies
```bash
pip install torch transformers datasets evaluate rouge-score nltk wordcloud seaborn scikit-learn
```

### Step 2 — Run notebooks in parallel (one per machine)

**Machine 1:**
```bash
jupyter nbconvert --to notebook --execute bonus_rnn_lstm_qa.ipynb
```

**Machine 2:**
```bash
jupyter nbconvert --to notebook --execute pipeline_BERT_DistilBART.ipynb
```

**Machine 3:**
```bash
jupyter nbconvert --to notebook --execute pipeline_T5_only.ipynb
```

### Step 3 — Load all saved models for inference

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM

qa_model    = AutoModelForQuestionAnswering.from_pretrained('./models_lg/qa_model')
sum_model   = AutoModelForSeq2SeqLM.from_pretrained('./models_lg/sum_model')
quiz_model  = AutoModelForSeq2SeqLM.from_pretrained('./models_lg/quiz_model')
```

> **Note:** All three `./models_lg/` directories must be synced to the same machine before running inference.

---

## Dependencies

```
python          >= 3.8
torch           >= 2.0
transformers    >= 4.40
datasets        >= 2.0
evaluate        >= 0.4
scikit-learn    >= 1.0
tensorflow      >= 2.12          # Only needed for Notebook 1 (RNN/LSTM)
keras           >= 2.12          # Only needed for Notebook 1
rouge-score     >= 0.1.2
nltk            >= 3.8
wordcloud       >= 1.9
matplotlib      >= 3.7
seaborn         >= 0.12
pandas          >= 2.0
numpy           >= 1.24
```

CUDA-enabled GPU is **strongly recommended**. All notebooks use `fp16=True` for mixed-precision training and auto-detect `cuda` vs `cpu`.

---

## Notes

- All random seeds are fixed at `random_state=42` for reproducibility.
- Early stopping patience is set to **2 epochs** across all fine-tuning jobs.
- The `valid_mask` mechanism in `QADataset` automatically skips samples where the answer string cannot be found in the context (character-level search), preventing index-out-of-range errors during tokenization.
- Checkpoint loading logic in `load_log_history()` falls back to the latest `checkpoint-N/trainer_state.json` if the root `trainer_state.json` is absent.
