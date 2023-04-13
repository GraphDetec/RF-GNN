import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
import torch
import torch.nn as nn
from models import RGCN, GAT, GCN, SAGE, SGC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Dataset import cresci15, Twibot20, MGTAB
from utils import sample_mask, init_weights
import numpy as np
import argparse
import time
import json
from config import Config

parser = argparse.ArgumentParser()
parser.add_argument("-dataset", type=str, default='Twibot20', help="dataset", choices=['Twibot20','MGTAB','Cresci15'])
parser.add_argument("-ensemble", type=bool, default=True, help="whether use ensemble")
parser.add_argument('-model', type=str, default='GCN', choices=['GCN', 'GAT', 'GraphSage', 'RGCN', 'SGC'])
parser.add_argument('--labelrate', type=float, default=0.1, help='labelrate')
args = parser.parse_args()
print(args)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
config_file = "./config/" + str(args.dataset) + ".ini"
config = Config(config_file)


if args.dataset == 'Twibot20':
    dataset = Twibot20('Data/Twibot20')
elif args.dataset == 'MGTAB':
    dataset = MGTAB('Data/MGTAB')
elif args.dataset == 'Cresci15':
    dataset = cresci15('Data/Cresci15')


data = dataset[0]
if args.dataset == 'MGTAB':
    data.y = data.y2

out_dim = 2
data = data.to(device)
sample_number = len(data.y)

index_select_list = (data.edge_type == 100)
relation_dict = {
    0:'followers',
    1:'friends'
}

relation_select_list = json.loads(config.relation_select)
relation_num = len(relation_select_list)
print('relation used:', end=' ')
for features_index in relation_select_list:
        index_select_list = index_select_list + (features_index == data.edge_type)
        print('{}'.format(relation_dict[features_index]), end='  ')
edge_index = data.edge_index[:, index_select_list]
edge_type = data.edge_type[index_select_list]


def main(seed):

    np.random.seed(seed)
    torch.manual_seed(seed)

    node_id = np.arange(data.num_nodes)
    np.random.shuffle(node_id)
    data.n_id = torch.arange(data.num_nodes)
    data.train_id = node_id[:int(data.num_nodes * args.labelrate)]
    data.val_id = node_id[int(data.num_nodes * 0.1):int(data.num_nodes * 0.2)]
    data.test_id = node_id[int(data.num_nodes * 0.2):]

    data.train_mask = sample_mask(data.train_id, sample_number)
    data.val_mask = sample_mask(data.val_id, sample_number)
    data.test_mask = sample_mask(data.test_id, sample_number)

    test_mask = data.test_mask
    train_mask = data.train_mask
    val_mask = data.val_mask

    fdim = data.x.shape[1]
    embedding_size = fdim

    results = torch.zeros(data.x.shape[0], out_dim).to(device)
    if args.ensemble:
        model_num = config.model_num
    else:
        model_num = 1


    for num in range(model_num):
        print('traning {}th model'.format(num + 1))
        if args.model == 'RGCN':
            model = RGCN(embedding_size, config.hidden_dimension, out_dim, relation_num, config.dropout).to(device)
        elif args.model == 'GCN':
            model = GCN(embedding_size, config.hidden_dimension, out_dim, relation_num, config.dropout).to(device)
        elif args.model == 'GAT':
            model = GAT(embedding_size, config.hidden_dimension, out_dim, relation_num, config.dropout).to(device)
        elif args.model == 'GraphSage':
            model = SAGE(embedding_size, config.hidden_dimension, out_dim, relation_num, config.dropout).to(device)
        elif args.model == 'SGC':
            model = SGC(embedding_size, config.hidden_dimension, out_dim, relation_num, config.dropout).to(device)
        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config.lr, weight_decay=config.weight_decay)

        model.apply(init_weights)


        for epoch in range(config.epochs):
            model.train()
            output = model(data.x, edge_index, edge_type)
            loss_train = loss(output[data.train_mask], data.y[data.train_mask])
            out = output.max(1)[1].to('cpu').detach().numpy()
            label = data.y.to('cpu').detach().numpy()
            acc_train = accuracy_score(out[train_mask], label[train_mask])
            acc_val = accuracy_score(out[val_mask], label[val_mask])
            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()
            if (epoch + 1)%100 == 0:
                print('Epoch: {:04d}'.format(epoch + 1),
                      'loss_train: {:.4f}'.format(loss_train.item()),
                      'acc_train: {:.4f}'.format(acc_train.item()),
                      'acc_val: {:.4f}'.format(acc_val.item()))

        model.eval()
        output = model(data.x, edge_index, edge_type)
        label = data.y.to('cpu').detach().numpy()
        out = output.max(1)[1].to('cpu').detach().numpy()
        acc_test = accuracy_score(out[test_mask], label[test_mask])
        f1 = f1_score(out[test_mask], label[test_mask], average='macro')
        precision = precision_score(out[test_mask], label[test_mask], average='macro')
        recall = recall_score(out[test_mask], label[test_mask], average='macro')
        print('acc_test {:.4f}'.format(acc_test),
              'f1_test: {:.4f}'.format(f1.item()),
              'precision_test: {:.4f}'.format(precision.item()),
              'recall_test: {:.4f}'.format(recall.item()))
        results = results + output
    results_out = results.max(1)[1].to('cpu').detach().numpy()
    acc_test = accuracy_score(results_out[test_mask], label[test_mask])
    f1 = f1_score(results_out[test_mask], label[test_mask], average='macro')
    precision = precision_score(results_out[test_mask], label[test_mask], average='macro')
    recall = recall_score(results_out[test_mask], label[test_mask], average='macro')

    return acc_test, precision, recall, f1




if __name__ == "__main__":

    t = time.time()
    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []

    for i, seed in enumerate(json.loads(config.random_seed)):
        print('traning {}th round'.format(i + 1))
        acc, precision, recall, f1 = main(seed)
        acc_list.append(acc * 100)
        precision_list.append(precision * 100)
        recall_list.append(recall * 100)
        f1_list.append(f1 * 100)
        print('Round:{:04d}'.format(i + 1),
              'acc_test {:.4f}'.format(acc),
              'f1_test: {:.4f}'.format(f1),
              'precision_test: {:.4f}'.format(precision),
              'recall_test: {:.4f}'.format(recall))
    print('acc:       {:.2f} + {:.2f}'.format(np.array(acc_list).mean(), np.std(acc_list)))
    print('precision: {:.2f} + {:.2f}'.format(np.array(precision_list).mean(), np.std(precision_list)))
    print('recall:    {:.2f} + {:.2f}'.format(np.array(recall_list).mean(), np.std(recall_list)))
    print('f1:        {:.2f} + {:.2f}'.format(np.array(f1_list).mean(), np.std(f1_list)))
    print('total time:', time.time() - t)