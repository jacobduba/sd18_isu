from data_processing import create_table, process_data, create_code_search_net_dataset

if __name__ == "__main__":
    create_table()
    data_points = create_code_search_net_dataset()
    if data_points:
        process_data(data_points)
