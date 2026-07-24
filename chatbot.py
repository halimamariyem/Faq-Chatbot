"""
FAQ Chatbot Engine
==================
CodeAlpha Internship - Task 2: Chatbot for FAQs

Pipeline:
 1. Collect FAQs (questions + answers) from a CSV file.
 2. Preprocess text using NLTK (tokenize, remove stopwords/punctuation, lemmatize).
 3. Match user questions to the most similar FAQ using TF-IDF + cosine similarity.
 4. Return the best matching answer (with a confidence score).

Run directly for a command-line chat:
    python chatbot.py
"""

import csv
import os
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------------------------
# 1. Setup: make sure required NLTK data is available (downloads once)
# ---------------------------------------------------------------------------
def ensure_nltk_data():
    """Download required NLTK corpora/models if they are not already present."""
    resources = {
        "tokenizers/punkt": "punkt",
        "tokenizers/punkt_tab": "punkt_tab",
        "corpora/stopwords": "stopwords",
        "corpora/wordnet": "wordnet",
        "corpora/omw-1.4": "omw-1.4",
    }
    for path, package in resources.items():
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


ensure_nltk_data()

LEMMATIZER = WordNetLemmatizer()
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    STOPWORDS = set()
PUNCT_TABLE = str.maketrans("", "", string.punctuation)


# ---------------------------------------------------------------------------
# 2. Text preprocessing
# ---------------------------------------------------------------------------
def preprocess(text: str) -> str:
    """
    Clean and normalize text:
      - lowercase
      - remove punctuation
      - tokenize
      - remove stopwords
      - lemmatize each token
    Returns a single cleaned string (tokens joined by spaces) ready for
    vectorization.
    """
    text = text.lower().strip()
    text = text.translate(PUNCT_TABLE)

    try:
        tokens = word_tokenize(text)
    except LookupError:
        # Fallback simple tokenizer if NLTK data is unavailable
        tokens = text.split()

    cleaned_tokens = [
        LEMMATIZER.lemmatize(tok)
        for tok in tokens
        if tok not in STOPWORDS and tok.strip() != ""
    ]
    return " ".join(cleaned_tokens)


# ---------------------------------------------------------------------------
# 3. The FAQ Chatbot class
# ---------------------------------------------------------------------------
class FAQChatbot:
    """
    Loads an FAQ dataset from CSV, preprocesses it, and matches new
    user queries to the closest FAQ question using TF-IDF + cosine similarity.
    """

    def __init__(self, csv_path: str, similarity_threshold: float = 0.25):
        self.csv_path = csv_path
        self.similarity_threshold = similarity_threshold
        self.questions = []
        self.answers = []
        self.vectorizer = None
        self.tfidf_matrix = None

        self._load_faqs()
        self._build_index()

    def _load_faqs(self):
        """Read questions & answers from the CSV file."""
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"FAQ file not found: {self.csv_path}")

        with open(self.csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = row.get("question", "").strip()
                a = row.get("answer", "").strip()
                if q and a:
                    self.questions.append(q)
                    self.answers.append(a)

        if not self.questions:
            raise ValueError("No FAQs loaded — check your CSV file content.")

    def _build_index(self):
        """Preprocess all FAQ questions and fit a TF-IDF vectorizer over them."""
        processed_questions = [preprocess(q) for q in self.questions]
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(processed_questions)

    def get_response(self, user_query: str):
        """
        Given a raw user query, return a dict with:
          - answer: best matching answer (or a fallback message)
          - matched_question: the FAQ question that matched
          - score: cosine similarity score (0-1)
        """
        if not user_query or not user_query.strip():
            return {
                "answer": "Please type a question so I can help you.",
                "matched_question": None,
                "score": 0.0,
            }

        processed_query = preprocess(user_query)
        query_vec = self.vectorizer.transform([processed_query])

        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        best_idx = similarities.argmax()
        best_score = float(similarities[best_idx])

        if best_score < self.similarity_threshold:
            return {
                "answer": (
                    "I'm sorry, I couldn't find a good match for that question. "
                    "Could you try rephrasing it, or ask something else related to the FAQs?"
                ),
                "matched_question": None,
                "score": best_score,
            }

        return {
            "answer": self.answers[best_idx],
            "matched_question": self.questions[best_idx],
            "score": best_score,
        }

    def top_matches(self, user_query: str, top_n: int = 3):
        """Return the top N candidate matches with scores (useful for debugging)."""
        processed_query = preprocess(user_query)
        query_vec = self.vectorizer.transform([processed_query])
        similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        ranked = sorted(
            zip(self.questions, self.answers, similarities),
            key=lambda x: x[2],
            reverse=True,
        )
        return ranked[:top_n]


# ---------------------------------------------------------------------------
# 4. Command-line chat interface
# ---------------------------------------------------------------------------
def run_cli():
    csv_path = os.path.join(os.path.dirname(__file__), "faqs.csv")
    bot = FAQChatbot(csv_path)

    print("=" * 60)
    print(" FAQ Chatbot (CodeAlpha Task 2) - type 'quit' to exit")
    print("=" * 60)

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ("quit", "exit", "bye"):
            print("Bot: Goodbye! 👋")
            break

        result = bot.get_response(user_input)
        print(f"Bot: {result['answer']}")
        if result["matched_question"]:
            print(f"     (matched: \"{result['matched_question']}\" "
                  f"| confidence: {result['score']:.2f})")


if __name__ == "__main__":
    run_cli()
