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


# Set up the OpenRouter client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),
)

if not client.api_key:
    raise ValueError("OpenRouter API key is missing. Set it in your environment variables.")

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

    print("\n--- Sending API Request ---")
    print(f"Query: {user_input}")
    print(f"Snippet: {snippet[:200]}...")  # Print first 200 chars of snippet
    print("--------------------------------------------------")

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                extra_headers={},
                model="deepseek/deepseek-r1",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            # Safely check if an error object is present
            error = getattr(completion, "error", None)
            if error:
                print(f"API Error (attempt {attempt + 1}): {error.get('message', 'Unknown error')}")
                continue  # skip this attempt

            # Check if choices exist
            if not completion.choices or not completion.choices[0].message:
                print(f"Missing choices or message on attempt {attempt + 1}")
                continue

            response_text = completion.choices[0].message.content

            if not response_text:
                print(f"Attempt {attempt + 1}: LLM returned empty response.")
                continue  # try next attempt

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
