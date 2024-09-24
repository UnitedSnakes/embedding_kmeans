import os
import pandas as pd

def print_csv_row_counts_recursively(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.csv'):
                file_path = os.path.join(root, file)
                try:
                    df = pd.read_csv(file_path)
                    row_count = len(df)
                    print(f"{file_path}: {row_count} rows")
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    folder_path = "results"
    print_csv_row_counts_recursively(folder_path)
