import os
import requests
from tqdm import tqdm

def download_amazon_review_data(file_name):
    """Download Amazon review data file if it does not exist in the current directory.

    Args:
        file_name (str): Name of the Amazon review data file.

    Returns:
        str: Notification about the file status.
    """
    url = f"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/{file_name}.csv"

    file_path = f'./dataset/{file_name}.csv'

    if os.path.exists(file_path):
        return f"File {file_name} already exists."

    response = requests.get(url, stream=True)
    if response.status_code == 200:
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte

        with open(file_path, 'wb') as file, tqdm(
            desc=file_name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                file.write(data)
                bar.update(len(data))

         # Add to .gitignore
        with open('./.gitignore', 'a') as gitignore:
            gitignore.write(f"\n{file_path[1:]}")
        return f"File {file_name} downloaded successfully."
    else:
        return f"Failed to download {file_name}. Error code: {response.status_code}"