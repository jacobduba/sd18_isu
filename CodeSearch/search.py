from search_processing import get_processed_data, process_user_code_segment, get_top_ten
import requests
from openai import OpenAI
import asyncio
import aiohttp
import os

# Set OpenRouter API Key & Base URL
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY"),  # Ensure this is set in your environment
)

if not client.api_key:
    raise ValueError("OpenRouter API key is missing. Set it in your environment variables.")

def evaluate_snippet(user_input: str, snippet: str):
    """ Sends one code snippet to DeepSeek via OpenRouter API and retrieves a score (0-10). """
    
    prompt = f"""
    You are an AI that scores code snippets based on their relevance to a given query.
    Score this snippet on a scale from 1 (completely unrelated) to 10 (perfect match).

    ### Query:
    {user_input}

    ### Code Snippet:
    {snippet}

    ### Scoring Criteria:
    1. **10**: The snippet **directly implements** the query and is highly efficient (e.g., full-function implementation for downloading videos).
    2. **8-9**: The snippet is **highly relevant**, but missing small components.
    3. **5-7**: The snippet is **partially relevant**, but does not fully meet the query.
    4. **1-4**: The snippet **mentions keywords** but lacks core functionality.
    5. **0**: The snippet is **completely unrelated**.

    ### Response Format:
    Return only a **single number between 0 and 10**, nothing else.
    """

    try:
        completion = client.chat.completions.create(
            extra_headers={},
            model="deepseek/deepseek-r1:free",  # DeepSeek model via OpenRouter
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )

        # Check if response exists before calling `.strip()`
        response_text = completion.choices[0].message.content if completion.choices and completion.choices[0].message.content else ""

        if not response_text:
            print("Warning: No response from LLM. Returning score 0.")
            return 0 # Default score if response is empty

        score = float(response_text.strip())  # Convert response to float
        return max(0, min(10, score))  # Ensure score is within 0-10

    except requests.exceptions.Timeout:
        print("Timeout Error: Skipping this snippet.")
        return 0 
    except (ValueError, IndexError, AttributeError):
        print("Error: Could not process snippet score. Returning 0.")
        return 0  # Default score if parsing fails

def rank_snippets(user_input: str, snippets: list):
    """ Evaluates each snippet separately and returns a list sorted by score. """
    
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
