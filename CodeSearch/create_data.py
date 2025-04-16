import sqlite3

from data_processing import (
    DB_FILE,
    create_code_search_net_dataset,
    create_table,
    process_data,
)

slice_size = 9
if __name__ == "__main__":
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        create_table(cursor)
        data_points = create_code_search_net_dataset(slice_size=slice_size)
        if data_points:
            process_data(data_points, cursor)
        cursor.execute("SELECT COUNT(*) FROM embeddings")
        print("Rows in DB:", cursor.fetchone()[0])
        conn.commit()
