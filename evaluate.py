import torch
import numpy as np
import random
from sklearn.metrics import average_precision_score
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, train_data, test_data, num_users, num_items, top_k, num_user_sample, num_neg_test_samples):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_data = train_data
        self.test_data = test_data
        self.num_users = num_users
        self.num_items = num_items
        self.top_k = top_k
        self.num_user_sample = num_user_sample
        self.num_neg_test_samples = num_neg_test_samples

    def evaluate(self):

         # Randomly sample a subset of users
        sampled_users = random.sample(range(self.num_users), self.num_user_sample)
        hit_rates, ndcgs = [], []

        for userid in tqdm(sampled_users, desc="Evaluating", total=len(sampled_users)):
            # Get the ground truth for the user
            user_data = self.test_data[self.test_data['userid_encoded'] == userid]
            positive_item = set(user_data[user_data['label'] == 1]['itemid_encoded'])

            # Get the ranked list with our test instance with the negative samples
            rank_list = self.get_ranklist(userid, list(positive_item))

            # Calculate metrics
            hit_rates.append(self.calculate_hit_rate(rank_list))
            ndcgs.append(self.calculate_ndcg(rank_list))

        # Aggregate results
        hit_rate = np.mean(hit_rates)
        ndcg = np.mean(ndcgs)


        return {
            'Hit Rate@K': hit_rate,
            'NDCG': ndcg,
        }

    def get_ranklist(self, userid, pos_items):
        self.model.eval()

        # Generate predictions for all items for the given user
        candidate_list = random.sample(list(set(range(self.num_items))
                                            -set(self.train_data[self.train_data['userid_encoded'] == userid]['itemid_encoded'])
                                            -set(self.test_data[self.test_data['userid_encoded'] == userid]['itemid_encoded'])), self.num_neg_test_samples)
        user_tensor = torch.tensor([userid] * (self.num_neg_test_samples + 1))  # Repeat the user ID for each item
        items_tensor = torch.tensor(pos_items + candidate_list)

        # Move tensors to the same device as the model
        user_tensor = user_tensor.to(self.device)
        items_tensor = items_tensor.to(self.device)

        with torch.no_grad():
            predictions = self.model(user_tensor, items_tensor).squeeze().cpu().tolist()

        return predictions

    # Helper methods for calculating metrics
    def calculate_hit_rate(self, rank_list):
        rank = sum(np.array(rank_list) >= rank_list[0])
        return int(rank <= self.top_k)

    def calculate_ndcg(self, rank_list):
        # Initialize the binary relevance list
        rank = sum(np.array(rank_list) >= rank_list[0])
        return 1 / np.log(rank + 1) if rank <= self.top_k else 0