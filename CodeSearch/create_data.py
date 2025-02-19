from data_processing import process_data, create_code_search_net_dataset

if __name__ == "__main__":
    data_points = create_code_search_net_dataset()
    if data_points:
        process_data(data_points)
