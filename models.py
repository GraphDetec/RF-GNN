import torch
from torch import nn
from torch_geometric.nn import RGCNConv,GCN2Conv,GCNConv,GATConv,SAGEConv,SGConv,RGATConv,GINConv
import torch.nn.functional as F



class RGCN(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(RGCN, self).__init__()
        self.dropout = dropout
        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.rgcn1 = RGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)
        self.rgcn2 = RGCNConv(hidden_dimension, hidden_dimension, num_relations=relation_num)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.rgcn1(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.rgcn1(x, edge_index, edge_type)
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x



class GAT(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(GAT, self).__init__()
        self.dropout = dropout

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

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.gat1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x



class SGC(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(SGC, self).__init__()
        self.dropout = dropout

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

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.gcn1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x



class GCN(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(GCN, self).__init__()
        self.dropout = dropout

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

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.gcn1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gcn2(x, edge_index)
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x



class SAGE(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(SAGE, self).__init__()
        self.dropout = dropout

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

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.sage1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.sage2(x, edge_index)
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x


class GIN(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(GIN, self).__init__()
        self.dropout = dropout

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )

        self.gin1 = GINConv(nn=nn.Linear(hidden_dimension, hidden_dimension), eps=1e-9)
        self.gin2 = GINConv(nn=nn.Linear(hidden_dimension, hidden_dimension), eps=1e-13)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        x = self.gin1(x, edge_index)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gin2(x, edge_index)
        # x = self.linear_relu_output1(x)
        x = self.linear_output2(x)

        return x



class GCN2(nn.Module):
    def __init__(self, embedding_dimension=16, hidden_dimension=128, out_dim=3, relation_num=2, dropout=0.3):
        super(GCN2, self).__init__()
        self.dropout = dropout

        self.linear_relu_input = nn.Sequential(
            nn.Linear(embedding_dimension, hidden_dimension),
            nn.LeakyReLU()
        )

        self.gcn1 = GCN2Conv(hidden_dimension, 0.2, add_self_loops=False)
        self.gcn2 = GCN2Conv(hidden_dimension, 0.2, add_self_loops=False)

        self.linear_relu_output1 = nn.Sequential(
            nn.Linear(hidden_dimension, hidden_dimension),
            nn.LeakyReLU()
        )
        self.linear_output2 = nn.Linear(hidden_dimension, out_dim)

    def forward(self, feature, edge_index, edge_type):
        x = self.linear_relu_input(feature.to(torch.float32))
        h = self.gcn1(x, x, edge_index)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.gcn2(h, x, edge_index)
        # x = self.linear_relu_output1(x)
        h = self.linear_output2(h)

        return h