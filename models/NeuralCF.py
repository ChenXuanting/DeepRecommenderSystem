import torch
import torch.nn as nn
import torch.nn.init as init

class NCF(nn.Module):
    def __init__(self, num_factors, num_users, num_items, nums_hiddens):
        super(NCF, self).__init__()
        self.P = nn.Embedding(num_users, num_factors)
        self.Q = nn.Embedding(num_items, num_factors)
        self.U = nn.Embedding(num_users, num_factors)
        self.V = nn.Embedding(num_items, num_factors)

        MLP_modules = []
        nums_hiddens.insert(0, 2*num_factors)
        for i in range(len(nums_hiddens)-1):
            linear_layer = nn.Linear(nums_hiddens[i], nums_hiddens[i + 1])

            # Apply He initialization
            init.kaiming_normal_(linear_layer.weight, nonlinearity='relu')

            MLP_modules.append(linear_layer)
            MLP_modules.append(nn.ReLU())
        self.mlp = nn.Sequential(*MLP_modules)

        self.prediction_layer = nn.Linear(num_factors + nums_hiddens[-1], 1, bias=False)
        init.kaiming_normal_(self.prediction_layer.weight, nonlinearity='relu')

    def forward(self, user_id, item_id):
        p_mf = self.P(user_id)
        q_mf = self.Q(item_id)
        gmf = p_mf * q_mf

        p_mlp = self.U(user_id)
        q_mlp = self.V(item_id)
        mlp_input = torch.cat([p_mlp, q_mlp], dim=1)
        mlp = self.mlp(mlp_input)

        con_res = torch.cat([gmf, mlp], dim=1)
        prediction = self.prediction_layer(con_res)
        return torch.relu(prediction)
