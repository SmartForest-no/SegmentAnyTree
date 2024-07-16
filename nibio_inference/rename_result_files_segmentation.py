import sys
import os
import yaml

from nibio_inference.bring_back_to_utm_coordinates import bring_back_to_utm_coordinates

def rename_files(yaml_file, directory):
    try:
        with open(yaml_file, 'r') as file:
            # Load the YAML file
            data = yaml.load(file, Loader=yaml.FullLoader)

            # Get the fold section
            fold_section = data.get('data', {}).get('fold', [])

            for index, file_path in enumerate(fold_section):
                # Extract the file name from the path
                file_name = os.path.basename(file_path)
                
                # Create new file name as result_index.ply
                old_file_name = f'semantic_result_{index}.ply'
                old_file_path = os.path.join(directory, old_file_name)

                # put semantic_segmentation_ in front of the file name
                new_file_name = 'semantic_segmentation_'+ file_name
                new_file_path = os.path.join(directory, new_file_name)

                # Rename the file
                os.rename(old_file_path, new_file_path)

                # bring_back_to_utm_coordinates(new_file_path, file_path)
                
                print(f'Renamed {old_file_path} to {new_file_path} ')

    except Exception as e:
        print(f'An error occurred: {e}')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python script.py <yaml_file> <directory>')
        sys.exit(1)
    
    yaml_file = sys.argv[1]
    directory = sys.argv[2]

    rename_files(yaml_file, directory)
