from search_processing import get_processed_data, process_user_code_segment, get_top_ten
import requests
from openai import OpenAI
import time
import os

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

# Set OpenRouter API Key & Base URL
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),  # Ensure this is set in your environment
)

if not client.api_key:
    raise ValueError("OpenRouter API key is missing. Set it in your environment variables.")

def evaluate_snippet(user_input: str, snippet: str, retries=3):
    """ Sends one code snippet to DeepSeek via OpenRouter API and retrieves a score (0-10). """

    prompt = SCORING_PROMPT_TEMPLATE.format(query=user_input, snippet=snippet)

    print("\n--- Sending API Request ---")
    print(f"Query: {user_input}")
    print(f"Snippet: {snippet[:200]}...")  # Print first 200 chars of snippet
    print("--------------------------------------------------")

    for attempt in range(retries):
        print(f"\n--- Sending API Request (Attempt {attempt + 1}) ---")
        print(f"Query: {user_input}")
        print(f"Snippet: {snippet[:200]}...")
        print("--------------------------------------------------")

        try:
            completion = client.chat.completions.create(
                extra_headers={},
                model="deepseek/deepseek-r1:free",  
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )

            # Print full API response
            print("\n--- API Response ---")
            print(completion)

            # Extract response text
            response_text = completion.choices[0].message.content if completion.choices and completion.choices[0].message.content else ""

            if not response_text:
                print(f"Warning: No response from LLM (Attempt {attempt + 1}). Retrying...")
                time.sleep(1)  # Small delay before retry to not overwhelm API
                continue

            # Convert response to float
            score = float(response_text.strip())
            print(f"Extracted Score: {score}")
            return max(0, min(10, score))

        except requests.exceptions.Timeout:
            print("Timeout Error: Retrying...")
            time.sleep(1)
            continue
        except (ValueError, IndexError, AttributeError) as e:
            print(f"Error: Could not process snippet score. Retrying... Error: {e}")
            time.sleep(1)
            continue

    print("Final Warning: No response from LLM after multiple attempts. Returning score 0.")
    return 0  # Default score if all attempts fail

def rank_snippets(user_input: str, snippets: list):
    """ Evaluates each snippet separately and returns a list sorted by score. """

    print("\n--- Ranking Snippets ---")
    scored_snippets = [(snippet, evaluate_snippet(user_input, snippet)) for snippet in snippets]

    # Sort by score (highest first)
    scored_snippets.sort(key=lambda x: x[1], reverse=True)

    return scored_snippets

if __name__ == "__main__":
    user_input = input("Enter your code description: ")

    # Get processed data (code snippets & embeddings)
    processed_data = get_processed_data()
    processed_user_code = process_user_code_segment(user_input)

    # Get top 10 most relevant snippets using vector similarity search
    top_ten_indices = get_top_ten(processed_user_code, processed_data)

    # Extract the actual snippets from the indices
    top_ten_snippets = [processed_data[index].code_string for index, _ in top_ten_indices]

    # Rank the top 10 snippets using the LLM API
    ranked_snippets = rank_snippets(user_input, top_ten_snippets)

    # Display the ranked results
    print("\nRanked Snippets:")
    for rank, (snippet, score) in enumerate(ranked_snippets, start=1):
        print(f"{rank} (Score: {score:.1f}):\n{snippet}\n{'-'*50}")
