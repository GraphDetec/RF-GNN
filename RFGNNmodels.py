import torch
from torch import nn
from torch_geometric.nn import RGCNConv,GCNConv,GATConv,SAGEConv,SGConv
import torch.nn.functional as F



class RGCN(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(RGCN, self).__init__()
        self.dropout = dropout
        self.out_dim = out_dim

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_relu_input2 = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.rgcn1 = RGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)
        self.rgcn2 = RGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output = nn.Linear(hidden_dimension, out_dim)
        self.linear_output22 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, mask_feature, feature, edge_index, edge_type):
        x = self.linear_relu_input(mask_feature.to(torch.float32))
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn2(x, edge_index, edge_type)
        # x = self.linear_relu_output1(x)
        x = self.linear_output(x)
        mask = self.linear_relu_input(feature-mask_feature)
        mask = self.rgcn1(mask, edge_index, edge_type)
        mask = self.rgcn2(mask, edge_index, edge_type)
        mask = self.linear_output(mask)
        aplha = torch.mul(x, mask).sum(1).repeat(self.out_dim,1)
        # return torch.mul(aplha.T, mask)
        return torch.mul(x, mask)



class GAT(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=2, relation_num=2, dropout=0.3):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.out_dim = out_dim

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )

        self.gat1 = GATConv(hidden_dimension, int(hidden_dimension / 4), heads=4)
        self.gat2 = GATConv(hidden_dimension, hidden_dimension)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, mask_feature, feature, edge_index, edge_type):
        x = self.linear_relu_input(mask_feature.to(torch.float32))
        x = self.gat1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        mask = self.linear_relu_input(feature - mask_feature)
        mask = self.linear_output2(mask)
        aplha = torch.mul(x, mask).sum(1).repeat(self.out_dim, 1)
        # return torch.mul(aplha.T, mask)
        return torch.mul(x, mask)



class GCN(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=2, relation_num=2, dropout=0.3):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.out_dim = out_dim

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )

        self.gcn1 = GCNConv(hidden_dimension, hidden_dimension)
        self.gcn2 = GCNConv(hidden_dimension, hidden_dimension)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)
        self.linear_relu_mask = nn.Linear(embedding_dimension, 1)

    def forward(self, mask_feature, feature, edge_index, edge_type):

        x = self.linear_relu_input(mask_feature.to(torch.float32))
        x = self.gcn1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.linear_output2(x)
        mask = self.linear_relu_input(feature - mask_feature)
        mask = self.linear_output2(mask)
        aplha = torch.mul(x, mask).sum(1).repeat(self.out_dim,1)
        # return torch.mul(aplha.T, mask)
        return torch.mul(x, mask)



class SGC(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=2, relation_num=2, dropout=0.3):
        super(SGC, self).__init__()
        self.dropout = dropout
        self.out_dim = out_dim

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )

        self.gcn1 = SGConv(hidden_dimension, hidden_dimension)
        self.gcn2 = SGConv(hidden_dimension, hidden_dimension)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)
        self.linear_relu_mask = nn.Linear(embedding_dimension, 1)

    def forward(self, mask_feature, feature, edge_index, edge_type):

        x = self.linear_relu_input(mask_feature.to(torch.float32))
        x = self.gcn1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        x = self.linear_output2(x)
        mask = self.linear_relu_input(feature - mask_feature)
        mask = self.linear_output2(mask)
        aplha = torch.mul(x, mask).sum(1).repeat(self.out_dim,1)
        # return torch.mul(aplha.T, mask)
        return torch.mul(x, mask)



class SAGE(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(SAGE, self).__init__()
        self.dropout = dropout
        self.out_dim = out_dim
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )

        self.sage1 = SAGEConv(hidden_dimension, hidden_dimension)
        self.sage2 = SAGEConv(hidden_dimension, hidden_dimension)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, mask_feature, feature, edge_index, edge_type):
        x = self.linear_relu_input(mask_feature.to(torch.float32))
        x = self.sage1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage2(x, edge_index)
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)
        mask = self.linear_relu_input(feature - mask_feature)
        mask = self.linear_output2(mask)
        aplha = torch.mul(x, mask).sum(1).repeat(self.out_dim,1)
        # return torch.mul(aplha.T, mask)
        return torch.mul(x, mask)