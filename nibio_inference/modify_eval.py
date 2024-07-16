import os
import argparse
import yaml
from collections import OrderedDict

def get_all_ply_paths(directory):
    """Get paths of all .ply files in the given directory"""
    ply_paths = []
    for root, directories, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.ply'):
                filepath = os.path.join(root, filename)
                ply_paths.append(filepath)
    return ply_paths

def modify_yaml(file_path, new_fold, output_dir_path=None):
    """Modify the YAML file"""
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Ensuring data is an OrderedDict
    data = OrderedDict(data)

    # Update the fold field
    data['data']['fold'] = list(new_fold)

    # Update the output_dir field
    if output_dir_path:
        data['hydra']['run']['dir'] = output_dir_path

    # Use this function to output an OrderedDict as YAML
    def ordered_dump(data, stream=None, Dumper=yaml.Dumper, **kwds):
        class OrderedDumper(Dumper):
            pass
        def _dict_representer(dumper, data):
            return dumper.represent_mapping(
                yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                data.items())
        OrderedDumper.add_representer(OrderedDict, _dict_representer)
        return yaml.dump(data, stream, OrderedDumper, **kwds)

    with open(file_path, 'w') as file:
        ordered_dump(data, file, Dumper=yaml.SafeDumper, default_flow_style=False)

def main():
    parser = argparse.ArgumentParser(description='Modify fold field in a YAML file.')
    parser.add_argument('yaml_file_path', help='Path to the YAML file to be modified.')
    parser.add_argument('folder_path', help='Path to the folder containing .ply files.')
    parser.add_argument('output_dir_path', help='Path to the output directory.')

    args = parser.parse_args()

    # Get all .ply file paths in the provided folder
    ply_paths = get_all_ply_paths(args.folder_path)
    if not ply_paths:
        print(f"No .ply files found in the directory: {args.folder_path}")
        return

    # Modify the YAML file
    modify_yaml(args.yaml_file_path, ply_paths, args.output_dir_path)

if __name__ == '__main__':
    main()
