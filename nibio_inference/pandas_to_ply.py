import numpy as np
import pandas as pd
from plyfile import PlyElement, PlyData

def pandas_to_ply(csv, csv_file_provided=False, output_file_path=None):

    

     # Check if the csv_file_provided argument is provided

    if csv_file_provided:
        df = pd.read_csv(csv)
    else:
        df = csv

    # remove duplicated columns
    df = df.loc[:,~df.columns.duplicated()]

    # Replace spaces in column names with underscores
    df.columns = [col.replace(' ', '_') for col in df.columns]
 
    # Create a structured numpy array with dtype based on the columns of the DataFrame
    dtypes = [(col, 'f4') for col in df.columns]
    data = np.array(list(map(tuple, df.to_records(index=False))), dtype=dtypes)

    # Create a new PlyElement
    vertex = PlyElement.describe(data, 'vertex')

    # Save the data to a PLY file
    if output_file_path is None:
        ply_file_path = output_file_path.replace('.csv', '.ply')
    
    ply_data = PlyData([vertex], text=False)
    ply_data.write(output_file_path)

if __name__ == "__main__":
    csv_path = "/path/to/your/file.csv"
    ply_path = "/path/to/your/file.ply"

    pandas_to_ply(csv_path, ply_path)
