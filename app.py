import os
from flask import Flask, render_template, request, jsonify
import requests
import pandas as pd
import json


app = Flask(__name__)

# ==== CONFIGURATION ====
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
df = pd.read_csv("customer_support_tickets.csv")

PRODUCTS = df["Product Purchased"].unique()

TICKET_TYPES = [
    "Refund request",
    "Technical issue",
    "Cancellation request",
    "Product inquiry",
    "Billing inquiry",
]


def groq_generate_response(
    product: str,
    ticket_type: str,
    ticket_subject: str,
    user_query: str,
) -> str:
    """
    Return a support reply plus 1-3 follow-up questions as a JSON-formatted string.
    """
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise EnvironmentError("GROQ_API_KEY not set")

    user_prompt = (
        f"Context:\n"
        f"- Product: {product}\n"
        f"- Ticket Type: {ticket_type}\n"
        f"- Ticket Subject: {ticket_subject}\n"
        f"- Customer Query: {user_query}\n\n"
        "Instructions:\n"
        "1. Provide a professional, empathetic answer with clear, actionable steps.\n"
        "2. Anticipate 1-3 follow-up questions the customer may ask.\n"
        "3. Respond **strictly in JSON** with this structure:\n"
        "{\n"
        '  "answer": "string",\n'
        '  "follow_up_questions": [\n'
        '     {"question": "string", "answer": "string"}\n'
        "  ]\n"
        "}\n"
        "Do not include markdown, code blocks, or any additional commentary."
    )


    payload = {
        "model": "llama3-8b-8192",
        "temperature": 0.5,
        "max_tokens": 300,
        "messages": [
            {
            "role": "system",
            "content": "You are a helpful customer-support assistant.",
            },
            {"role": "user", "content": user_prompt},
        ],
    }


    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=30,
    )
    if not r.ok:
        raise RuntimeError(f"Groq API error {r.status_code}: {r.text}")

    raw_content = r.json()["choices"][0]["message"]["content"]
    try:
        return json.loads(raw_content)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Invalid JSON from model: {e}\nRaw output:\n{raw_content}")



@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", products=PRODUCTS, ticket_types=TICKET_TYPES)


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    product = data.get("product")
    ticket_type = data.get("ticket_type")
    ticket_subject = data.get("ticket_subject")
    user_msg = data.get("user_msg")
    bot_reply = groq_generate_response(product, ticket_type, ticket_subject, user_msg)
    return jsonify(bot_reply)


if __name__ == "__main__":
    app.run(debug=True)
