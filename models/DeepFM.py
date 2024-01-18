import torch
import torch.nn as nn

class DFM(nn.Module):
    def __init__(self, feature_sizes, embedding_size=5, hidden_dims=[256, 128], num_class=1, dropout=0.5):
        super(DFM, self).__init__()
        self.num_class = num_class

        # FM Part
        # First-order embeddings
        self.fm_first_order_embeddings = nn.ModuleList([nn.Embedding(size, 1) for size in feature_sizes])
        # Second-order embeddings
        self.fm_second_order_embeddings = nn.ModuleList([nn.Embedding(size, embedding_size) for size in feature_sizes])

        # Deep Part
        all_dims = [len(feature_sizes) * embedding_size] + hidden_dims + [1]
        self.deep_layers = nn.ModuleList()
        for i in range(len(all_dims)-1):
            self.deep_layers.append(nn.Linear(all_dims[i], all_dims[i+1]))
            self.deep_layers.append(nn.BatchNorm1d(all_dims[i+1]))
            self.deep_layers.append(nn.ReLU())
            self.deep_layers.append(nn.Dropout(dropout))

        self.final = nn.Linear(3, num_class)

    def forward(self, *args):
        """
        Forward pass
        X: Input tensor, shape (batch_size, num_fields)
        """
        X = torch.stack([field.detach() for field in args], dim=1)

        # FM First Order
        fm_first_order = torch.cat([emb(X[:, i]) for i, emb in enumerate(self.fm_first_order_embeddings)], 1)
        fm_first_order_out = torch.sum(fm_first_order, 1)

        # FM Second Order
        fm_second_order = [emb(X[:, i]) for i, emb in enumerate(self.fm_second_order_embeddings)]
        fm_second_order_sum_square = torch.sum(torch.stack(fm_second_order), 0) ** 2
        fm_second_order_square_sum  = torch.sum(torch.stack([item** 2 for item in fm_second_order]), 0)
        fm_second_order_out = torch.sum((fm_second_order_sum_square - fm_second_order_square_sum) * 0.5, 1)

        # Deep Part
        deep_emb = torch.cat(fm_second_order, 1)
        deep_out = deep_emb
        for i in range(0, len(self.deep_layers), 4):  # Process layers in steps of 4 (linear, batchnorm, relu, dropout)
            deep_out = self.deep_layers[i](deep_out)
            deep_out = self.deep_layers[i+1](deep_out)
            deep_out = self.deep_layers[i+2](deep_out)
            deep_out = self.deep_layers[i+3](deep_out)

        # Final prediction
        out = torch.stack((fm_first_order_out, fm_second_order_out, deep_out.squeeze()), 1)
        out = self.final(out)
        out = torch.sigmoid(out)

        return out
