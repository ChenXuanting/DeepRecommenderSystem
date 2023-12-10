import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from tqdm import tqdm

class NCFDataset(Dataset):
    """PyTorch Dataset for loading data."""
    def __init__(self, df):
        self.users = torch.LongTensor(df['userid_encoded'].values)
        self.items = torch.LongTensor(df['itemid_encoded'].values)
        self.ratings = torch.FloatTensor(df['rating'].values)

    def __len__(self):
        return len(self.ratings)

    def __getitem__(self, idx):
        return self.users[idx], self.items[idx], self.ratings[idx]

def train_model(model, train_loader, num_epochs, learning_rate):
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move model to GPU if available
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")
        for i, (users, items, ratings) in progress_bar:
            # Move data to GPU if available
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(users, items).squeeze()

            # Calculate loss
            loss = criterion(outputs, ratings)

            # Backward pass and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({'Loss': running_loss / (i + 1)})

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')