import sys
from dataset.download_data import download_amazon_review_data
from dataset.data_preprocessor import preprocess
from train import train_model
from models.NeuralCF import NCF
from models.DeepFM import DFM

if len(sys.argv) > 2:
    raise ValueError("Only one argument is accepted.")
elif len(sys.argv) < 2:
    raise ValueError("You have to provide the model name.")

#modify the ratings name variable to change dataset
ratings_name = "ratings_MovieLens_100k"
downloaded_data = download_amazon_review_data(ratings_name)
print(downloaded_data)
processed_train, processed_test, num_users, num_items  = preprocess(ratings_name, negative_sampling=True, num_neg=1)

if sys.argv[1] == "NCF":
    # Create model instance
    model = NCF(5, num_users, num_items, nums_hiddens= [128, 64, 32, 16])
elif sys.argv[1] == 'DFM':
    model = DFM([num_users, num_items])
else:
    raise ValueError("The model name you provided is not recognized.")

# Train the model
train_model(model, processed_train, processed_test, num_users, num_items,
            top_k=10, num_user_sample=min(10000, num_users), num_neg_test_samples=100,
            num_epochs=20, batch_size = 32, learning_rate=0.01)
