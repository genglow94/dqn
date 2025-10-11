import torch
import torch.nn as nn 

class QNet(nn.Module):#nn.Module:PyTorch が提供してくれる基本クラス
    def __init__(self, obs_dim, n_actions,hidden_dim=64): #abs_dim:状態の次元
        super().__init__()
        self.fc = nn.Sequential(#fc:全結合層（Fully Connected Layer）
             nn.Linear(obs_dim,hidden_dim),
             nn.ReLU(),
             nn.Linear(hidden_dim,hidden_dim),
             nn.ReLU(),
             nn.Linear(hidden_dim,n_actions)
        )

    def forward(self,x):
        return self.fc(x)