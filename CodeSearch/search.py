from search_processing import get_processed_data, process_user_code_segment, get_top_ten

if __name__ == "__main__":
    user_input = input("Enter your code description: ")
    processed_data = get_processed_data()
    processed_user_code = process_user_code_segment(user_input)
    top_ten = get_top_ten(processed_user_code, processed_data)
    counter = 1
    for index, score in top_ten:
        print(f"{counter}: Similarity score: {score} \n Comment string: {processed_data[index].comment_string} \n Code snippet: {processed_data[index].code_string}")
        counter += 1
