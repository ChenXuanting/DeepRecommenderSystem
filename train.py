import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from evaluate import Evaluator

class NCFDataset(Dataset):
    """PyTorch Dataset for loading data."""
    def __init__(self, df):
        self.users = torch.LongTensor(df['userid_encoded'].values)
        self.items = torch.LongTensor(df['itemid_encoded'].values)
        self.labels = torch.FloatTensor(df['label'].values)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.labels[idx]

def train_model(model, train_data, test_data, num_users, num_items, top_k, num_user_sample, num_epochs, batch_size, learning_rate):
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to GPU if available
    model = model.to(device)

    train_dataset = NCFDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for i, (users, items, labels) in progress_bar:
            # Move data to GPU if available
            users, items, labels = users.to(device), items.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(users, items).squeeze()

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({'Loss': running_loss / (i + 1)})

        evaluator = Evaluator(model, train_data, test_data, num_users, num_items, top_k, num_user_sample)
        metrics = evaluator.evaluate()
        print(f'Hit rate@K: {metrics["Hit Rate@K"]}, NDCG: {metrics["NDCG"]}, MAP: {metrics["MAP"]}, Precision@K: {metrics["Precision@K"]}, Recall@K: {metrics["Recall@K"]}')