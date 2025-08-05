import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class transformer_net(nn.Module):
    def __init__(self, d_input, d_model, num_heads, num_layers, d_output, num_partons, position_embedding=True, dropout=0.1 ):
        super(transformer_net, self).__init__()

        self.d_input = d_input
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.d_output = d_output
        self.num_partons = num_partons
        self.position_embedding = position_embedding
        self.dropout = dropout

        self.input_proj = nn.Linear(self.d_input, self.d_model)
        if not self.position_embedding:
            self.id_embedding = nn.Embedding(self.num_partons, self.d_model)

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.num_heads,
            dim_feedforward=self.num_heads*self.d_model,
            dropout=self.dropout,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        if self.position_embedding:
            self.output_head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.d_model * self.num_partons, 128),
                nn.ReLU(),
                nn.Linear(128, self.d_output)  # d_output = number of swap actions
            )
        else:
            self.output_head = nn.Sequential(
                nn.Linear(self.d_model, 128),
                nn.ReLU(),
                nn.Linear(128, self.d_output)  # d_output = number of swap actions
            )

    def forward(self, x):
        x = x.transpose(1, 2)
        # x: (B, N, 4)
        B, N, _ = x.shape
        # (B, N, d_model) 
        x_embed = self.input_proj(x)  # (B, N, d_model)
        if not self.position_embedding:
            # add particle ID embedding (hardcoded ordering IDs)
            ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)  # (B, N)
            id_embed = self.id_embedding(ids)
            x_embed = x_embed + id_embed

        # (B, N, d_model)
        x_trans = self.transformer(x_embed)

        if not self.position_embedding:
            x_trans = x_trans.sum(1)

        return self.output_head(x_trans)

    def save(self, path):
        torch.save(self.state_dict(), path)