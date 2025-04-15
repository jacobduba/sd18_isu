from flask import Flask, render_template, request
from search_processing import get_processed_data, process_user_code_segment, get_top_ten
from openai import OpenAI
import requests
import time
import os

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# Scoring prompt for LLM
SCORING_PROMPT_TEMPLATE = """
You are an AI evaluating code snippets.

- Your task: Score this snippet's relevance to the query.
- **Return ONLY a number between 0-10**.
- **DO NOT** explain your reasoning.
- **DO NOT** include any text, words, punctuation, or comments.
- Example response: `8`

### Query:
{query}

### Code Snippet:
{snippet}

### Response Format:
- Output **only** a single integer between `0` and `10`.
- No additional text.
- If unsure, provide your **best numerical estimate**.
"""

def evaluate_snippet(user_input: str, snippet: str, retries=3):
    prompt = SCORING_PROMPT_TEMPLATE.format(query=user_input, snippet=snippet)
    #ollama api is hosted locally on port 11434
    url = "http://localhost:11434/api/generate"
    print("\n--- Sending API Request ---")
    print(f"Query: {user_input}")
    print(f"Snippet: {snippet[:200]}...")  # Print first 200 chars of snippet
    print("--------------------------------------------------")

    for attempt in range(retries):
        try:
            completion = requests.post(url, json={
                "model":"llama2",
                "prompt":prompt,
                "stream":False,
                "temperature":0.2,
            })

            # Safely check if an error object is present

            if completion.status_code != 200:
                print(f"API Error (attempt {attempt + 1}): {completion.text}")
                continue  # skip this attempt
            
            data = completion.json()
            response_text = data.get("response","")

            # Check if choices exist
            if not response_text:
                print(f"Missing choices or message on attempt {attempt + 1}")
                continue


            # Print full API response
            print("\n--- API Response ---")
            print(completion)

            score = float(response_text.strip())
            return max(0, min(10, score))

        except Exception as e:
            print(f"Exception on attempt {attempt + 1}: {e}")
            time.sleep(1)

    print("All attempts failed. Returning score 0.")
    return 0


def rank_snippets(user_input: str, snippets: list):
    scored_snippets = [(snippet, evaluate_snippet(user_input, snippet)) for snippet in snippets]
    scored_snippets.sort(key=lambda x: x[1], reverse=True)
    return scored_snippets


@app.route("/", methods=["GET", "POST"])
def search_page():
    results = None
    if request.method == "POST":
        user_input = request.form["code_description"]
        processed_data = get_processed_data()
        processed_user_code = process_user_code_segment(user_input)
        top_ten_indices = get_top_ten(processed_user_code, processed_data)
        top_ten_snippets = [processed_data[index].code_string for index, _ in top_ten_indices]
        ranked_snippets = rank_snippets(user_input, top_ten_snippets)
        results = ranked_snippets

    return render_template("index.html", results=results)

if __name__ == "__main__":
    app.run(debug=True)
