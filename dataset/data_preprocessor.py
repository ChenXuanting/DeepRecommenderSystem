import pandas as pd
from tqdm import tqdm
import random

def preprocess(file_name, negative_sampling = False, num_neg=1):
    # Path to the CSV file
    file_path = f'./dataset/{file_name}.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path, names=['userid', 'itemid', 'rating', 'timestamp'])

    # Assuming positive interactions are those with a rating (implicit feedback)
    df_positives = df[['userid', 'itemid', 'timestamp']].drop_duplicates()
    df_positives['label'] = 1

    num_users = df_positives['userid'].nunique()
    num_items = df_positives['itemid'].nunique()
    num_records = df_positives.shape[0]
    print(f'Matrix sparsity: {1-num_records/(num_users*num_items)}')

    if negative_sampling:
        # Negative sampling
        all_items = list(df['itemid'].unique())
        negatives = []
        for userid, group in tqdm(df.groupby('userid'), desc='Negative Sampling'):
            pos_items = set(group['itemid'])
            neg_samples_count = num_neg * len(pos_items)
            added_negs = 0
            added_neg_items = []

            while (added_negs < neg_samples_count) and (added_negs + len(pos_items) < num_items):
                neg_item = all_items[random.randint(0, len(all_items) - 1)]
                if (neg_item not in pos_items) and (neg_item not in added_neg_items):
                    negatives.append([userid, neg_item, 0])
                    added_neg_items.append(neg_item)
                    added_negs += 1

        df_negatives = pd.DataFrame(negatives, columns=['userid', 'itemid', 'label'])

        # Combine positives and negatives
        df_combined = pd.concat([df_positives, df_negatives])
    else:
        df_combined = df_positives

    # Encode 'userid' and 'itemid'
    user_encoder = {uid: idx for idx, uid in enumerate(df_combined['userid'].unique())}
    item_encoder = {iid: idx for idx, iid in enumerate(df_combined['itemid'].unique())}

    df_combined['userid_encoded'] = df_combined['userid'].map(user_encoder)
    df_combined['itemid_encoded'] = df_combined['itemid'].map(item_encoder)

    # Leave-one-out split
    df_combined['rank_latest'] = df_combined.groupby(['userid_encoded'])['timestamp'].rank(method='first', ascending=False)
    train_df = df_combined[df_combined['rank_latest'] != 1]
    test_df = df_combined[df_combined['rank_latest'] == 1]

    return train_df[['label', 'userid_encoded', 'itemid_encoded']], test_df[['label', 'userid_encoded', 'itemid_encoded']], num_users, num_items