import torch
import numpy as np
import random
from sklearn.metrics import average_precision_score
from tqdm import tqdm

class Evaluator:
    def __init__(self, model, train_data, test_data, num_users, num_items, top_k, num_user_sample):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.train_data = train_data
        self.test_data = test_data
        self.num_users = num_users
        self.num_items = num_items
        self.top_k = top_k
        self.num_user_sample = num_user_sample

    def evaluate(self):

         # Randomly sample a subset of users
        random.seed(42)
        sampled_users = random.sample(range(self.num_users), self.num_user_sample)
        hit_rates, ndcgs, maps, precisions, recalls = [], [], [], [], []

        for userid in tqdm(sampled_users, desc="Evaluating", total=len(sampled_users)):
            # Get the ground truth for the user
            user_data = self.test_data[self.test_data['userid_encoded'] == userid]
            positive_items = set(user_data[user_data['label'] == 1]['itemid_encoded'])

            # Get top K recommendations for the user
            top_k_recommendations = set(self.recommend(userid))

            # Calculate metrics
            hit_rates.append(self.calculate_hit_rate(top_k_recommendations, positive_items))
            ndcgs.append(self.calculate_ndcg(top_k_recommendations, positive_items))
            maps.append(average_precision_score(user_data['label'], user_data['itemid_encoded'].isin(top_k_recommendations)))
            precisions.append(self.calculate_precision(top_k_recommendations, positive_items))
            recalls.append(self.calculate_recall(top_k_recommendations, positive_items))

        # Aggregate results
        hit_rate = np.mean(hit_rates)
        ndcg = np.mean(ndcgs)
        map_score = np.mean(maps)
        precision_at_k = np.mean(precisions)
        recall_at_k = np.mean(recalls)


        return {
            'Hit Rate@K': hit_rate,
            'NDCG': ndcg,
            'MAP': map_score,
            'Precision@K': precision_at_k,
            'Recall@K': recall_at_k
        }

    def recommend(self, userid):
        self.model.eval()

        # Generate predictions for all items for the given user
        candidate_list = list(set(range(self.num_items))-set(self.train_data[self.train_data['userid_encoded'] == userid]['itemid_encoded']))
        user_tensor = torch.tensor([userid] * len(candidate_list))  # Repeat the user ID for each item
        items_tensor = torch.tensor(candidate_list)

        # Move tensors to the same device as the model
        user_tensor = user_tensor.to(self.device)
        items_tensor = items_tensor.to(self.device)

        with torch.no_grad():
            predictions = self.model(user_tensor, items_tensor).squeeze()

        # Get the top K items with the highest scores
        _, top_k_indices = torch.topk(predictions, self.top_k)

        # Convert to a list of item IDs
        top_k_indices = top_k_indices.cpu().tolist()

        return [candidate_list[index] for index in top_k_indices]

    # Helper methods for calculating metrics
    def calculate_hit_rate(self, top_k_recommendations, positive_items):
        return int(len(top_k_recommendations & positive_items) > 0)

    def calculate_ndcg(self, top_k_recommendations, positive_items):
        # Initialize the binary relevance list
        relevance = [1 if item in positive_items else 0 for item in top_k_recommendations]

        # Calculate DCG (Discounted Cumulative Gain)
        dcg = sum(rel / np.log2(idx + 2) for idx, rel in enumerate(relevance))

        # Calculate IDCG (Ideal Discounted Cumulative Gain)
        idcg = sum(1 / np.log2(idx + 2) for idx in range(min(len(positive_items), len(top_k_recommendations))))

        # Calculate NDCG
        ndcg = dcg / idcg if idcg > 0 else 0

        return ndcg

    def calculate_precision(self, top_k_recommendations, positive_items):
        return len(top_k_recommendations & positive_items) / len(top_k_recommendations)

    def calculate_recall(self, top_k_recommendations, positive_items):
        return len(top_k_recommendations & positive_items) / len(positive_items) if positive_items else 0