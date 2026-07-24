"""
Flask Web App for the FAQ Chatbot
==================================
CodeAlpha Internship - Task 2 (Optional: simple chat UI)

Run:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

import os
from flask import Flask, request, jsonify, render_template

from chatbot import FAQChatbot

app = Flask(__name__)

CSV_PATH = os.path.join(os.path.dirname(__file__), "faqs.csv")
bot = FAQChatbot(CSV_PATH)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_message = data.get("message", "")

    result = bot.get_response(user_message)
    return jsonify(
        {
            "answer": result["answer"],
            "matched_question": result["matched_question"],
            "score": round(result["score"], 3),
        }
    )


@app.route("/api/faqs", methods=["GET"])
def list_faqs():
    """Returns the full FAQ list, useful for showing suggested questions in the UI."""
    return jsonify(
        [{"question": q, "answer": a} for q, a in zip(bot.questions, bot.answers)]
    )


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
