import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split

def preprocess(file_name, num_neg=1):
    # Path to the CSV file
    file_path = f'./dataset/{file_name}.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, names=['userid', 'itemid', 'rating', 'timestamp'])

    # Assuming positive interactions are those with a rating (implicit feedback)
    df_positives = df[['userid', 'itemid', 'timestamp']].drop_duplicates()
    df_positives['label'] = 1

    # Negative sampling
    all_items = list(df['itemid'].unique())
    negatives = []
    for userid, group in tqdm(df.groupby('userid'), desc='Negative Sampling'):
        pos_items = set(group['itemid'])
        neg_samples_count = num_neg * len(pos_items)
        added_negs = 0

        while added_negs < neg_samples_count:
            neg_item = all_items[random.randint(0, len(all_items) - 1)]
            if neg_item not in pos_items:
                negatives.append([userid, neg_item, 0])
                added_negs += 1

    df_negatives = pd.DataFrame(negatives, columns=['userid', 'itemid', 'label'])

    # Combine positives and negatives
    df_combined = pd.concat([df_positives, df_negatives]).fillna(0)

    # Encode 'userid' and 'itemid'
    user_encoder = {uid: idx for idx, uid in enumerate(df_combined['userid'].unique())}
    item_encoder = {iid: idx for idx, iid in enumerate(df_combined['itemid'].unique())}

    df_combined['userid_encoded'] = df_combined['userid'].map(user_encoder)
    df_combined['itemid_encoded'] = df_combined['itemid'].map(item_encoder)

    # Leave-one-out split
    df_combined['rank_latest'] = df_combined.groupby(['userid_encoded'])['timestamp'].rank(method='first', ascending=False)
    train_df = df_combined[df_combined['rank_latest'] != 1].drop(columns=['timestamp', 'userid', 'itemid'])
    test_df = df_combined[df_combined['rank_latest'] == 1].drop(columns=['timestamp', 'userid', 'itemid'])

    return train_df, test_df, df_combined['userid_encoded'].nunique(), df_combined['itemid_encoded'].nunique()