import os
import sqlite3
import time

from datasets.arrow_dataset import re
from flask import Flask, render_template, request
from openai import OpenAI

from transformers import pipeline

from data_processing import DB_FILE
from search_processing import get_processed_data, get_top_ten, process_user_code_segment

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

generator = pipeline("text-generation", model="gpt2")

def evaluate_snippet(user_input: str, snippet: str, retries=3):
    """Evaluates a snippet locally using a text generation model to score its relevance."""
    prompt = SCORING_PROMPT_TEMPLATE.format(query=user_input, snippet=snippet)
    
    for attempt in range(retries):
        print(f"\n--- Local Model Inference (Attempt {attempt + 1}) ---")
        print(f"Prompt (truncated): {prompt[:200]}...")
        
        try:
            # The max_length should be set to limit the output to a few tokens.
            result = generator(prompt, max_new_tokens=10, temperature=0.2)
            generated_text = result[0]['generated_text']
            print("\n--- Model Output ---")
            print(generated_text)
            
            # We expect only a number: try extracting the first integer between 0 and 10.
            match = re.search(r'\b([0-9]|10)\b', generated_text)
            if match:
                score = float(match.group(1))
                print(f"Extracted Score: {score}")
                return max(0, min(10, score))
            else:
                print(f"Warning: Unable to find score in output (Attempt {attempt + 1}). Retrying...")
                time.sleep(1)
                continue
        
        except Exception as e:
            print(f"Error during local inference (Attempt {attempt + 1}): {e}")
            time.sleep(1)
            continue

    print("Final Warning: No valid score after multiple attempts. Returning score 0.")
    return 0  # Default fallback score


def rank_snippets(user_input: str, snippets: list):
    scored_snippets = [(snippet, evaluate_snippet(user_input, snippet)) for snippet in snippets]
    scored_snippets.sort(key=lambda x: x[1], reverse=True)
    return scored_snippets


@app.route("/", methods=["GET", "POST"])
def search_page():
    initial_results = None
    llm_results = None
    if request.method == "POST":
        with sqlite3.connect(DB_FILE) as conn:
            cursor = conn.cursor()
            user_input = request.form["code_description"]
            processed_data = get_processed_data(cursor)
            processed_user_code = process_user_code_segment(user_input)
            top_ten_indices = get_top_ten(processed_user_code, processed_data)

            initial_results = [(processed_data[index][0], score) for index, score in top_ten_indices]
            top_ten_snippets = [snippet for snippet, _ in initial_results]
            llm_results = rank_snippets(user_input, top_ten_snippets)

    return render_template("index.html", initial_results=initial_results, llm_results=llm_results)

if __name__ == "__main__":
    app.run(debug=True)
