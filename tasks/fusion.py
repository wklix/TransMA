import torch

import torch.nn as nn
import torch.nn.functional as F
class MyModule(nn.Module):
    def __init__(self, device=None):
        super(MyModule, self).__init__()
        self.device = device
        #self.bn = nn.LayerNorm(normalized_shape=[1024]).to(self.device)
        #self.merged_features = []
        self.fc = nn.Sequential(
            nn.Linear(1024, 1024 // 2).to(self.device),
            nn.ReLU(),
            nn.Linear(1024 // 2, 1).to(self.device)
        )
        self.sigmoid = nn.Sigmoid()
        self.gate_layer = nn.Sequential(
            nn.Linear(1024, 512).to(self.device),
            nn.Tanh(),
            #nn.Dropout(0.1),  # 添加Dropout
            #nn.Linear(512, 128).to(self.device),
            #nn.ReLU(),
            nn.Dropout(0.1),  # 添加Dropout
            nn.Linear(512, 1).to(self.device),
            nn.Tanh()
        )
    def forward(self, merged_features):
       
        attention_scores = self.fc(merged_features)
        attention_weights = self.sigmoid(attention_scores)

        weighted_feature = merged_features * attention_weights
        x_pool = torch.mean(weighted_feature, dim=0)
        value= self.gate_layer(x_pool)
        return value,attention_weights,x_pool
        