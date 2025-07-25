import pickle
import pandas as pd
from os import listdir
from os.path import isfile, join

BASE_PATH = 'full_pkl_files'
SAVE_PATH = 'test_pkl_files'

# Get the list of pkl files
pkl_files = [f for f in listdir(BASE_PATH) if isfile(join(BASE_PATH, f))]
print(pkl_files)

# Open the csv with test queries
df = pd.read_csv("easyjailbreak/datasets/harmbench_dataset_test.csv")
print(len(df))

def save_dict_to_pkl(data, filename):
    """
    Saves the given dictionary data to a specified pickle (.pkl) file.
    """
    try:
        # Attempt to open the file and write the pickle data
        with open(filename, 'wb') as file:
            pickle.dump(data, file)
        return True
    except (pickle.PicklingError, TypeError) as e:
        # Handles serialization errors if the data is not pickle serializable
        print(f"Error: Provided data is not pickle serializable - {e}")
    except IOError as e:
        # Handles file writing errors
        print(f"Error: Could not write to file '{filename}' - {e}")

    return False  # Return False if an error occurs

def print_pkl_file(file_path="attackPrompts_dict.pkl"):
    """
    Function to quickly print the pkl file
    """
    with open(file_path, 'rb') as file:
        loaded_dict = pickle.load(file)
        loaded_dict = dict(list(loaded_dict.items())[:2])

        total_count = 0
        for key in loaded_dict:
            print(f"Query: {key}")

            for instance in loaded_dict[key][:1]:
                print(instance)
                total_count += 1

        print(f"Total queries: {total_count}")

# for curr_file in pkl_files:
#     with open(join(BASE_PATH, curr_file), 'rb') as file:
#         loaded_dict = pickle.load(file)
#         loaded_dict = dict(list(loaded_dict.items()))
#
#         new_dict = {}
#         count = 0
#         for query in df['query'].unique():
#             if query in loaded_dict.keys():
#                 new_dict[query] = loaded_dict[query]
#                 count += 1
#
#         print(f"Total queries for {curr_file}: {count}")
#         save_dict_to_pkl(new_dict, join(SAVE_PATH, curr_file))
