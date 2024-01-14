from dataset.download_data import download_amazon_review_data
from dataset.data_preprocessor import preprocess
from train import train_model
from models.NeuralCF import NCF

#modify the ratings name variable to change dataset
ratings_name = "ratings_MovieLens_100k"
downloaded_data = download_amazon_review_data(ratings_name)
print(downloaded_data)
processed_train, processed_test, num_users, num_items  = preprocess(ratings_name, negative_sampling=True, num_neg=1)

# Hyperparameters
num_factors = 5
nums_hiddens = [128, 64, 32, 16, 8]

# Create model instance
ncf = NCF(num_factors, num_users, num_items, nums_hiddens)

# Train the model
train_model(ncf, processed_train, processed_test, num_users, num_items,
            top_k=10, num_user_sample=min(100000, num_users), num_neg_test_samples=100,
            num_epochs=20, batch_size = 32, learning_rate=0.01)
