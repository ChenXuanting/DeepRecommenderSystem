import pandas as pd

def preprocess(file_name):
    # Path to the CSV file
    file_path = f'./dataset/{file_name}.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, names=['userid', 'itemid', 'rating', 'timestamp'])

    # Encode 'userid' and 'itemid'
    user_encoder = {uid: idx for idx, uid in enumerate(df['userid'].unique())}
    item_encoder = {iid: idx for idx, iid in enumerate(df['itemid'].unique())}

    df['userid_encoded'] = df['userid'].map(user_encoder)
    df['itemid_encoded'] = df['itemid'].map(item_encoder)

    df.drop(columns=['timestamp', 'userid', 'itemid'], inplace=True)

    return df