# FAQ Chatbot — CodeAlpha Task 2

A chatbot that answers user questions by matching them against a bank of FAQs
using NLP preprocessing and cosine similarity.

## How it satisfies the task

| Requirement | Where it's done |
|---|---|
| Collect FAQs (Q&A pairs) | `faqs.csv` — 18 sample Q&A pairs (edit/extend freely) |
| Preprocess text with NLP (NLTK) | `chatbot.py → preprocess()` — lowercasing, punctuation removal, tokenization, stopword removal, lemmatization |
| Match user questions to FAQs | `chatbot.py → FAQChatbot` — TF-IDF vectorization + cosine similarity |
| Display best matching answer | `get_response()` returns the top match; CLI and web UI both display it |
| Optional: simple chat UI | `app.py` + `templates/index.html` — a Flask web chat interface |

## Project structure

```
faq_chatbot/
├── faqs.csv              # FAQ dataset (question, answer columns)
├── chatbot.py             # Core NLP + matching engine (run for CLI chatbot)
├── app.py                 # Flask web server (run for the browser chat UI)
├── templates/
│   └── index.html         # Chat UI (HTML/CSS/JS, calls the Flask API)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

The first run will automatically download the small NLTK datasets it needs
(punkt tokenizer, stopwords, wordnet) — this requires internet access once.

## Usage

### Option A — Command-line chatbot

```bash
python chatbot.py
```

```
FAQ Chatbot (CodeAlpha Task 2) - type 'quit' to exit
You: how do I apply for the internship?
Bot: You can apply for an internship by visiting the official CodeAlpha
     website and filling out the internship application form...
     (matched: "How do I apply for an internship?" | confidence: 1.00)
```

### Option B — Web chat UI (optional bonus)

```bash
python app.py
```

Then open **http://127.0.0.1:5000** in your browser. Type a question or tap
one of the suggested questions.

## How the matching works

1. **Preprocessing** — every FAQ question (and every user query) is lowercased,
   stripped of punctuation, tokenized, cleared of stopwords ("the", "is",
   "a", ...), and lemmatized ("running" → "run") using NLTK.
2. **Vectorization** — all cleaned FAQ questions are converted into TF-IDF
   vectors with scikit-learn's `TfidfVectorizer`.
3. **Matching** — the user's cleaned query is vectorized with the same
   vocabulary, and its **cosine similarity** is computed against every FAQ
   question vector.
4. **Response** — the FAQ with the highest similarity score is returned as
   the answer, as long as the score clears a minimum confidence threshold
   (default `0.25`). Below that, the bot admits it doesn't have a good match.

## Customizing

- **Add more FAQs**: just add rows to `faqs.csv` — no code changes needed.
- **Adjust match strictness**: change `similarity_threshold` when creating
  `FAQChatbot(csv_path, similarity_threshold=0.25)` in `chatbot.py` / `app.py`.
- **Swap in spaCy instead of NLTK**: replace the `preprocess()` function's
  internals with `nlp(text)` calls — the rest of the pipeline (TF-IDF +
  cosine similarity) stays the same.
