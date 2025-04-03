from flask import Flask, render_template, request
from search import get_processed_data, process_user_code_segment, get_top_ten, rank_snippets

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = None
    if request.method == 'POST':
        user_input = request.form['code_description']

        # Core logic using search.py methods
        processed_data = get_processed_data()
        processed_user_code = process_user_code_segment(user_input)
        top_ten_indices = get_top_ten(processed_user_code, processed_data)
        top_ten_snippets = [processed_data[index].code_string for index, _ in top_ten_indices]
        ranked_snippets = rank_snippets(user_input, top_ten_snippets)

        results = ranked_snippets

    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
