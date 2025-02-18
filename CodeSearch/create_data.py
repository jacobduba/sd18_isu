from usecases import unixcoder

if __name__ == "__main__":
    data_points = unixcoder.create_code_search_net_dataset()
    if data_points:
        unixcoder.process_data(data_points)
