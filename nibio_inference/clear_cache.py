import argparse
import os
import yaml

parser = argparse.ArgumentParser(description='Read eval.yaml location')
parser.add_argument('--eval_yaml', type=str, required=True, help='Path to eval.yaml file')
args = parser.parse_args()

with open(args.eval_yaml) as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

# go to checkpoint_dir and '.hydra' folder and open overrides.yaml and find data.dataroot there

with open(os.path.join(data['checkpoint_dir'], '.hydra/overrides.yaml')) as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

for item in data:
    if 'data.dataroot' in item:
        path = item.split('=')[1].strip()
      
# add processed_0.2_test to the path
path = os.path.join(path, 'treeinsfused')
path = os.path.join(path, 'processed_0.2_test')

print(f"Clearing: {path}")

# clear all the files in the directory if the path exists
if os.path.exists(path):
    for filename in os.listdir(path):
        filepath = os.path.join(path, filename)
        if os.path.isfile(filepath):
            os.remove(filepath)
            print(f"Removed: {filepath}")

else:
    print(f"Path does not exist: {path}")
