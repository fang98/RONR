# -*- coding: utf-8 -*-

import warnings
import random
import numpy as np
import scipy.sparse as sp
import torch
from torch import optim
import torch.nn.functional as F
from torch import nn
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, accuracy_score,average_precision_score
from functools import partial
from torch_geometric.data import DataLoader
from torch_scatter import scatter_add, scatter_mean,scatter_sum
from torch_geometric.data import Data
from torch_geometric.nn import global_sort_pool,global_max_pool,global_mean_pool,global_add_pool
from torch_geometric.utils import remove_self_loops, add_self_loops
import networkx as nx
import os


def subgraph2data(ebunch,A,X,neighborhoods):
    u,v,sl = ebunch
    
    u_nei = neighborhoods[u][0] | neighborhoods[u][1] | neighborhoods[u][2] | neighborhoods[u][3]
    u_nei.discard(v)
    v_nei = neighborhoods[v][0] | neighborhoods[v][1] | neighborhoods[v][2] | neighborhoods[v][3]
    v_nei.discard(u)
    V_K = [u,v]+list(u_nei | v_nei)
    V_K = np.array(V_K)
    
    if X is None:
        x = np.random.random([len(V_K),0])
    else:
        x = X[V_K,:]
    
    n_nodes = np.size(x,0)
    
    dict_vk = {w:i for i,w in enumerate(V_K)}
    x1 = np.zeros((n_nodes,26))
    common_neighbors = []
    for i in range(4):
        for j in range(4):
            cn_nei = neighborhoods[u][i] & neighborhoods[v][j]
            for w in cn_nei:
                x1[dict_vk[w]][8+i*4+j] = 1
                common_neighbors.append(w)
    no_common_neighbors = set(np.setdiff1d(V_K,np.array(common_neighbors+[u,v])))
    for i in range(4):
        nei = neighborhoods[u][i]
        for w in nei:
            if w in no_common_neighbors:
                x1[dict_vk[w]][i] = 1
    for i in range(4):
        nei = neighborhoods[v][i]
        for w in nei:
            if w in no_common_neighbors:
                x1[dict_vk[w]][i+4] = 1
    x1[0,-2] = 1
    x1[1,-1] = 1
    
    x = np.hstack((x1,x))
    
    a = A[V_K,:][:,V_K]
    a[0,1] = 0
    tmp = sp.coo_matrix(a)
    sign = tmp.data
    ind_pos = np.where(sign>0)[0]
    ind_neg = np.where(sign<0)[0]
    row_p = tmp.row[ind_pos]
    col_p = tmp.col[ind_pos]
    row_n = tmp.row[ind_neg]
    col_n = tmp.col[ind_neg]
    edge_index = np.array([row_p,col_p])
    edge_index = torch.tensor(edge_index, dtype=torch.long)
    edge_index_neg = np.array([row_n,col_n])
    edge_index_neg = torch.tensor(edge_index_neg, dtype=torch.long)
    x = torch.tensor(x, dtype=torch.float)
    data = Data(x=x, edge_index=edge_index)
    data.edge_index_neg = edge_index_neg
    data.y = torch.LongTensor([max(sl,0)])
    
    return data


def graph2data(edgelist,A,X, neighborhoods):
    partial_worker = partial(subgraph2data,A=A,X=X,neighborhoods=neighborhoods)
    data_features = []
    for u,v,s in edgelist:
        res = partial_worker((u,v,s))
        data_features.append(res)
    return data_features


def evaluate(model,loader,r_pos,device):
    model.eval()
    all_targets = []
    all_scores = []
    
    for batch in loader:
        batch.to(device)
        out,_ = model(batch)
        all_scores.append(F.softmax(out,dim=1)[:, 1].cpu().detach())
        all_targets.extend(batch.y.tolist())
    all_scores = torch.cat(all_scores).cpu().numpy()
    y_pred11 = np.zeros(np.size(all_scores))
    y_pred11[np.where(all_scores>r_pos)] = 1
    upt_res = [0]*11
    for k in range(np.size(y_pred11)):
        if y_pred11[k]==1:
            if all_targets[k]==1:
                upt_res[7] += 1
            else:
                upt_res[8] += 1
        else:
            if all_targets[k]==0:
                upt_res[9] += 1
            else:
                upt_res[10] += 1
    upt_res[0] = roc_auc_score(all_targets,all_scores)
    upt_res[1] = f1_score(all_targets,y_pred11, average='macro')
    upt_res[2] = f1_score(all_targets,y_pred11, average='micro')
    upt_res[3] = f1_score(all_targets,y_pred11)
    upt_res[4] = precision_score(all_targets,y_pred11)
    upt_res[5] = recall_score(all_targets,y_pred11)
    upt_res[6] = accuracy_score(all_targets,y_pred11)
    return upt_res


def calculating_mean_std(auc):
    mean = np.mean(auc,1)*100
    std = np.std(auc,1)*100
    return mean, std



def get_graph(adj):
    G = nx.DiGraph()
    G.add_nodes_from(list(range( np.size(adj,0) )))
    tmp = sp.coo_matrix(adj)
    row,col,data = tmp.row,tmp.col,tmp.data
    
    for u,v,s in zip(row,col,data):
        G.add_edge(u, v, weight = s)
        
    return G


def get_neighbors(num_nodes,adj):
    G2 = get_graph(adj)
    neighborhoods = []
    for u in range(num_nodes):
        u_out_pos = set(i for i in G2.successors(u) if G2[u][i]['weight']==1)
        u_out_neg = set(i for i in G2.successors(u) if G2[u][i]['weight']==-1)
        u_in_pos = set(i for i in G2.predecessors(u) if G2[i][u]['weight']==1)
        u_in_neg = set(i for i in G2.predecessors(u) if G2[i][u]['weight']==-1)
        neighborhoods.append([u_out_pos,u_out_neg,u_in_pos,u_in_neg])
    return neighborhoods


class sub_sumgnn(torch.nn.Module):
    def __init__(self, attri_dim, hid_dim,hid_dim2,n_layers,is_self):
        super(sub_sumgnn, self).__init__()
        self.n_layers = n_layers
        self.is_self = is_self
        
        self.sum_att = nn.ModuleList()
        for i in range(n_layers*4):
            self.sum_att.append(nn.Linear(2*hid_dim,1))
        
        self.lin = nn.ModuleList()
        self.lin.append(nn.Linear(attri_dim,hid_dim))
        for i in range(1,n_layers):
            self.lin.append(nn.Linear(hid_dim,hid_dim))
        
        self.lin_concat = nn.ModuleList()
        for i in range(n_layers):
            self.lin_concat.append(nn.Linear(4*hid_dim,hid_dim))
        
        self.lin_sign = nn.Sequential(
            nn.Linear(4*hid_dim,hid_dim),
            nn.ReLU(),#ReLU Tanh
            nn.Linear(hid_dim,hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim,hid_dim2),
            nn.ReLU(),
            nn.Linear(hid_dim2,2))
        self.lin_direct = nn.Sequential(
            nn.Linear(4*hid_dim,hid_dim),
            nn.ReLU(),#ReLU tanh
            nn.Linear(hid_dim,2))
        
    def forward(self, data):
        data.to(device)
        x = data.x
        edge_index_list = [data.edge_index,data.edge_index_neg]
        
        idx1 = x[:,24]==1
        idx2 = x[:,25]==1
        
        concat_emb = []
        z = x
        for i in range(self.n_layers):
            z = self.sum_attention(self.lin_concat[i],self.sum_att[i*4:(i+1)*4],
                          z,edge_index_list,self.lin[i],False)
            z = F.tanh(z)
            concat_emb.append(z)
        z = torch.cat(concat_emb,1)
        
        x12 = torch.cat((z[idx1],z[idx2]),1)
        pred_sign = self.lin_sign(x12)
        
        x21 = torch.cat((z[idx2],z[idx1]),1)
        x12 = self.lin_direct(x12)
        x21 = self.lin_direct(x21)
        pred_direct = torch.cat((x12,x21),0)
        
        return pred_sign, pred_direct
    
    def sum_attention(self,lin_cat,lin_att,x, edge_index_list,lin,is_self=False):
        xx = []
        if is_self:
            xx.append(x)
        x = lin(x)
        
        for i in range(len(edge_index_list)):
            edge_index, _ = remove_self_loops(edge_index_list[i], None)
            row, col = edge_index
            
            ee = torch.cat((x[row],x[col]),1)
            att_score = lin_att[2*i](ee)
            att_score = F.tanh(att_score)
            att_score = torch.exp(att_score)
            ee = x[col]*att_score
            out = scatter_sum(ee, row, dim=0, dim_size=x.size(0))
            xx.append(out)
            
            ee = torch.cat((x[col],x[row]),1)
            att_score2 = lin_att[2*i+1](ee)
            att_score2 = F.tanh(att_score2)
            att_score2 = torch.exp(att_score2)
            ee = x[row]*att_score2
            out = scatter_sum(ee, col, dim=0, dim_size=x.size(0))
            xx.append(out)
        
        x = torch.cat(xx,1)
        x = lin_cat(x)
        return x


if __name__=='__main__':
    pool = None
    edgepath = ['soc-sign-bitcoinalpha.csv','soc-sign-bitcoinotc.csv','wiki-RfA.txt',
                'soc-sign-Slashdot090221.txt','soc-sign-epinions.txt']
    edgepath = ['soc-sign-bitcoinalpha.csv','soc-sign-bitcoinotc.csv']
    
    n = 5
    seed = [i for i in range(1,n+1)]
    count = []
    count = np.loadtxt('input/count.txt').astype(int)
    
    auc = np.zeros((len(edgepath),n))
    f1 = np.zeros((len(edgepath),n))
    accuracy = np.zeros((len(edgepath),n))
    precision = np.zeros((len(edgepath),n))
    recall = np.zeros((len(edgepath),n))
    f1_micro = np.zeros((len(edgepath),n))
    f1_macro = np.zeros((len(edgepath),n))
    tp = np.zeros((len(edgepath),n))
    tn = np.zeros((len(edgepath),n))
    fp = np.zeros((len(edgepath),n))
    fn = np.zeros((len(edgepath),n))
    mean = np.zeros((len(edgepath),4))
    std = np.zeros((len(edgepath),4))
    warnings.filterwarnings('ignore')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    epochs = 40
    lr = 0.001
    batch_size = 256
    upd = 5
    r_val= 0.05
    r_pos = 0.5
    hid_dim = 16
    hid_dim2 = 8
    lambda_loss = 0.1
    n_layers = 2
    is_self = False
    
    SEED = 42
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    
    if not os.path.exists('./subgnn_model'):
        os.makedirs('./subgnn_model')
    
    for i_data,datapath in enumerate(edgepath):
        for i in range(n):
            print('dataset :',datapath,'; loop num :',str(i+1))
            if i_data!=2:
                data_name = datapath[9:-4]
            else:
                data_name = datapath[:-4]
            
            split_seed = seed[i]
            train_path = 'input/'+datapath+'_'+'train'+'_'+str(split_seed)+'.txt'
            train_edges = np.loadtxt(train_path).astype(int)
            test_path = 'input/'+datapath+'_'+'test'+'_'+str(split_seed)+'.txt'
            test_edges = np.loadtxt(test_path).astype(int)
            
            train_edges[:,:2] = train_edges[:,:2]-1
            test_edges[:,:2] = test_edges[:,:2]-1
            num_nodes = count[i_data][0]
            
            if r_val>0:
                n_train = train_edges.shape[0]
                idx = np.random.permutation(n_train)
                n_val = int(n_train*r_val)
                n_train = n_train-n_val
                val_edges = train_edges[idx[:n_val],:]
                train_edges = train_edges[idx[n_val:],:]
                
            
            adj_train = sp.csr_matrix((train_edges[:,2].astype(float), (train_edges[:,0], train_edges[:,1])),
                           shape = (num_nodes, num_nodes))
            
            
            X = None
            n_feas = 26
            if X is not None:
                n_feas = n_feas+X.shape[1]
            
            neighborhoods = get_neighbors(num_nodes,adj_train)
            train_features = graph2data(train_edges,adj_train,X, neighborhoods)
            test_features = graph2data(test_edges,adj_train,X, neighborhoods)
            train_loader = DataLoader(train_features, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_features, batch_size=batch_size, shuffle=False)
            if r_val>0:
                val_features = graph2data(val_edges,adj_train,X, neighborhoods)
                val_loader = DataLoader(val_features, batch_size=batch_size, shuffle=False)
            
            
            model = sub_sumgnn(n_feas, hid_dim, hid_dim2, n_layers, is_self)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            model.to(device)
            loss_func = nn.CrossEntropyLoss()
            
            n_samples = 0
            best_res = 0
            for epoch in range(epochs):
                model.train()
                total_loss = []
                
                for batch in train_loader:
                    out,direct = model(batch)
                    
                    loss = loss_func(out, batch.y)
                    y_direct = torch.LongTensor([1]*batch.y.shape[0]+[0]*batch.y.shape[0]).to(device)
                    loss += lambda_loss*loss_func(direct, y_direct)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    total_loss.append( loss.item() * len(batch.y))
                    n_samples += len(batch.y)
                    
                    
                total_loss = np.array(total_loss)
                avg_loss = np.sum(total_loss, 0) / n_samples
                
                if (epoch + 1) % upd == 0:
                    print(epoch+1, avg_loss)
                    upt_res = evaluate(model,test_loader,r_pos,device)
                    print('test metrics :',upt_res[:4])
                    if r_val>0:
                        upt_res = evaluate(model,val_loader,r_pos,device)
                        print('val metrics :',upt_res[:4])
                    
                    if upt_res[0]+upt_res[1] > best_res and r_val>0:
                        torch.save(obj=model.state_dict(), f='subgnn_model/'+data_name+'_'+str(i+1)+'.pth')
                        best_res = upt_res[0]+upt_res[1]
                        print('saving model in epoch =',epoch+1)
                    
                    print('**************************')
                    
            if r_val>0:
                new_model = sub_sumgnn(n_feas, hid_dim, hid_dim2, n_layers, is_self).to(device)
                new_model.load_state_dict(torch.load('subgnn_model/'+data_name+'_'+str(i+1)+'.pth'))
                upt_res = evaluate(new_model,test_loader,r_pos,device)
            else:
                upt_res = evaluate(model,test_loader,r_pos,device)
            auc[i_data][i],f1_macro[i_data][i],f1_micro[i_data][i],f1[i_data][i] = \
                upt_res[0],upt_res[1],upt_res[2],upt_res[3]
            
            torch.cuda.empty_cache()
            
            
            mean[:,0], std[:,0] = calculating_mean_std(f1_micro)
            mean[:,1], std[:,1] = calculating_mean_std(f1)
            mean[:,2], std[:,2] = calculating_mean_std(f1_macro)
            mean[:,3], std[:,3] = calculating_mean_std(auc)
