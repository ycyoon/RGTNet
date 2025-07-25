import csv

def read_csv1_as_tuples(file_path):
    """
    Reads a CSV file and returns a list of rows as tuples.
    Assumes the first row is the header.
    """
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = []
        for row in reader:
            rows.append(row['Behavior'])
    return rows

def read_csv2_as_tuples(file_path):
    """
    Reads a CSV file and returns a list of rows as tuples.
    Assumes the first row is the header.
    """
    with open(file_path, mode='r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows_query = []
        rows_responses = []
        for row in reader:
            rows_query.append(row['query'])
            rows_responses.append(row['reference_response'])
    return rows_query, rows_responses

def find_common_indices(file1, file2):
    """
    Finds and returns a list of indices from file1 where the row is also present in file2.
    """
    # Read both CSV files
    rows1 = read_csv1_as_tuples(file1)  # Convert to set for faster lookup
    (rows2_query, rows2_responses) = read_csv2_as_tuples(file2) 
    rows1 = set(rows1)

    # Find indices where row in file1 is also in file2 (indices correspond to indexing in file2)
    common_indices = [index for index, row in enumerate(rows2_query) if row in rows1]

    output_result = zip(rows2_query, rows2_responses)
    filtered_result = [item for i, item in enumerate(output_result) if i in common_indices]
    filtered_result.insert(0, ("query", "reference_response"))
    print(filtered_result)
    with open("harmbench_dataset_test.csv", "w") as csv_file:
        writer = csv.writer(csv_file)

        writer.writerows(filtered_result)
        

    return common_indices

if __name__ == "__main__":
    # Replace 'file1.csv' and 'file2.csv' with your actual file paths
    file1_path = 'easyjailbreak/datasets/harmbench_dataset_test.csv'
    file2_path = 'easyjailbreak/datasets/harmbench_dataset.csv'

    common_indices = find_common_indices(file1_path, file2_path)
    print(common_indices)
    print(len(common_indices))
