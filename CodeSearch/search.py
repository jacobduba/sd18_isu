from flask import Flask, render_template, request
from search_processing import get_processed_data, process_user_code_segment, get_top_ten
from openai import OpenAI
import requests
import time
import os

app = Flask(
    __name__,
    template_folder="../templates",     # relative path to templates
    static_folder="../static"           # relative path to static
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

    for attempt in range(retries):
        try:
            completion = client.chat.completions.create(
                extra_headers={},
                model="deepseek/deepseek-r1:free",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            response_text = completion.choices[0].message.content
            if not response_text:
                raise ValueError("Empty response from LLM.")
            score = float(response_text.strip())
            return max(0, min(10, score))
        except Exception as e:
            print(f"Error on attempt {attempt + 1}: {e}")
            time.sleep(1)

    return 0  # Fallback score

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
