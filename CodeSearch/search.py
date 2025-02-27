from search_processing import get_processed_data, process_user_code_segment, get_top_ten

if __name__ == "__main__":
    user_input = input("Enter your code description: ")
    code_segment_list, embedding_list, comment_string_list = get_processed_data()
    processed_user_code = process_user_code_segment(user_input)
    top_ten = get_top_ten(processed_user_code, embedding_list)
    counter = 1
    for index, score in top_ten:
        print(f"{counter}: Similarity score: {score} \n Comment string: {comment_string_list[index]} \n Code snippet: {code_segment_list[index]}")
        counter += 1
