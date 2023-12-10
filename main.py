from dataset.download_data import download_amazon_review_data
from dataset.data_preprocessor import preprocess
from train import train_model, NCFDataset
from torch.utils.data import DataLoader
from models.NeuralCF import NCF

#modify the ratings name variable to change dataset
ratings_name = "ratings_Home_and_Kitchen"
downloaded_data = download_amazon_review_data(ratings_name)
processed_data = preprocess(ratings_name)

train_dataset = NCFDataset(processed_data)
train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=True)

# Hyperparameters
num_factors = 50
num_users = processed_data['userid_encoded'].nunique()
num_items = processed_data['itemid_encoded'].nunique()
nums_hiddens = [128, 64, 32, 16, 8]

# Create model instance
ncf = NCF(num_factors, num_users, num_items, nums_hiddens)

# Train the model
train_model(ncf, train_loader, num_epochs=10, learning_rate=0.01)
