import os
import argparse
import random
import re

def rename_files(input_folder):
    # List all files in the given directory
    for filename in os.listdir(input_folder):
        # generate as many random numbers as the number of files in the directory
        # and put them in a list
        random_numbers = random.sample(range(1000, 9999), len(os.listdir(input_folder)))

        # Construct the file path
        filepath = os.path.join(input_folder, filename)
        # Check if it's a file
        if os.path.isfile(filepath):
            # Replace '-' with '_' in the filename
            # generate a random four digit number and append it to the filename in the beginning
            # as the string
            new_filename = str(random_numbers.pop()) + '_' + filename
            # new_filename = filename
            # # remove all the spaces from filename
            # # replace all the special characters from filename but not the extension
            # new_filename = re.sub(r'[^a-zA-Z0-9_\.]', '', new_filename)
            # # replace multiple underscores with single underscore
            # new_filename = re.sub(r'_+', '_', new_filename)
            new_filename = filename.replace('-', '_')
            new_filename = new_filename.replace(' ', '_')
        
            # Construct the new file path
            new_filepath = os.path.join(input_folder, new_filename)
            # Rename the file
            os.rename(filepath, new_filepath)
            print(f"Renamed: {filepath} to {new_filepath}")

def main():
    parser = argparse.ArgumentParser(description='Replace "-" with "_" in filenames within a directory.')
    parser.add_argument('input_folder', type=str, help='Path to the input folder')

    args = parser.parse_args()
    rename_files(args.input_folder)

if __name__ == '__main__':
    main()
