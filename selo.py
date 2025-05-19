# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.neural_network import MLPClassifier
import warnings
import scipy.sparse as sp
import math


def subgraph2vec(edge,K,A,ratio,alpha1,alpha2,alpha3):
    u,v = edge
    V_K = np.array([u,v])
    fringe = np.array([u,v])
    nodes_dist = np.array([1,1])
    dist = 1
    while np.size(V_K)<K and np.size(fringe)>0:
        nei = np.array([]).astype(int)
        nei_out = sp.find(A[fringe,:])[1]
        nei_in = sp.find(A[:,fringe])[0]
        nei = np.concatenate((nei,nei_out,nei_in))
        nei = np.unique(nei)
        fringe = np.setdiff1d(nei,V_K)
        V_K = np.concatenate((V_K,fringe))
        dist = dist+1
        nodes_dist = np.concatenate(( nodes_dist,dist*np.ones(fringe.shape[0]) ))
        
    V_K = V_K.astype(int)
    a = A[V_K,:][:,V_K].toarray()
    
    a[0][1] = 0
    a[np.where(a<0)] = -ratio
    for k in range(np.size(a,0)):
        a[k,k:] = a[k,k:]/nodes_dist[k]
        a[k:,k] = a[k:,k]/nodes_dist[k]
    num_sub_nodes = np.size(a,0)
    
    z1 = alpha1*np.linalg.inv(alpha1*a.T.dot(a)+np.eye(num_sub_nodes)).dot(a.T).dot(a)
    z2 = alpha2*np.linalg.inv(alpha2*a.dot(a.T)+np.eye(num_sub_nodes)).dot(a).dot(a)
    z3 = alpha3*a.dot(a).dot(np.linalg.inv(alpha3*a.T.dot(a)+np.eye(num_sub_nodes)))
    s1 = a.dot(z1)
    s2 = a.T.dot(z2)
    s3 = z3.dot(a.T)
    
    s = [s1,s2,s3]
    n_groups = len(s)
    
    orders = []
    for i in range(n_groups):
        tmp = np.abs(s[i])
        score = tmp[2:,0]+tmp[2:,1]+tmp[0,2:]+tmp[1,2:]
        orderi = np.concatenate(( np.array([0,1]),np.flip(np.argsort(score))+2 ))
        orders.append(orderi)
    
    if num_sub_nodes>=K:
        res = []
        for i in range(n_groups):
            res.append(s[i][orders[i][:K],:][:,orders[i][:K]])
    else:
        res = []
        for i in range(n_groups):
            tmp = np.zeros([K,K])
            tmp[:num_sub_nodes,:][:,:num_sub_nodes] = s[i][orders[i],:][:,orders[i]]
            res.append(tmp)
    for i in range(n_groups):
        res[i] = res[i].reshape(-1)
    
    selo = np.hstack(res)
    
    return selo


def graph2vec(edges,K,A,ratio,alpha1,alpha2,alpha3):
    features = []
    labels = []
    num = 0
    for u,v,s in edges:
        fea = subgraph2vec([u,v],K,A,ratio,alpha1,alpha2,alpha3)
        features.append(fea)
        labels.append(max(s,0))
        num += 1
        if num%1000==0:
            print(num)
    
    features = np.array(features)
    labels = np.array(labels)
    return features,labels


def feature_and_label(data_idx,datapath,split_seed,count,K,alpha1,alpha2,alpha3):
    train_edges = np.load('input/'+datapath[:-4]+'_train_'+str(split_seed)+'.npy',allow_pickle=True)#.astype(int)
    test_edges = np.load('input/'+datapath[:-4]+'_test_'+str(split_seed)+'.npy',allow_pickle=True)#.astype(int)
    
    num_nodes = count[data_idx][0]
    ratio = math.log10(np.size(np.where(train_edges[:,2]==1)[0])/np.size(np.where(train_edges[:,2]==-1)[0]))+1
    del_ind = [i*K+i for i in range(K)]+[i*K+i+K*K for i in range(K)]+\
        [i*K+i+K*K+K*K for i in range(K)]
    
    adj_train = sp.csr_matrix((train_edges[:,2].astype(float), (train_edges[:,0], train_edges[:,1])),
                    shape = (num_nodes, num_nodes))
    train_features,train_labels = graph2vec(train_edges,K,adj_train,ratio,alpha1,alpha2,alpha3)
    test_features,test_labels = graph2vec(test_edges,K,adj_train,ratio,alpha1,alpha2,alpha3)
    train_features = np.delete(train_features,del_ind,1)
    test_features = np.delete(test_features,del_ind,1)
    
    return train_features,train_labels,test_features,test_labels


def mlp(train_features,train_labels,test_features,test_labels):
    clf = MLPClassifier(hidden_layer_sizes=(32, 32, 16), alpha=1e-2,
                          batch_size=512, learning_rate_init=0.001,
                          max_iter=100,early_stopping=False, tol=-10000)
    clf.fit(train_features, train_labels)
    y_pred_proba = clf.predict_proba(test_features)[:,1]
    y_pred = clf.predict(test_features)
    
    auc = roc_auc_score(test_labels,y_pred_proba)
    f1 = f1_score(test_labels,y_pred)
    f1_micro = f1_score(test_labels,y_pred, average='micro')
    f1_macro = f1_score(test_labels,y_pred, average='macro')
    
    return f1_micro,f1,f1_macro,auc




if __name__=='__main__':
    warnings.filterwarnings('ignore')
    
    
    data_idx = 0
    alpha1 = 0.005
    alpha2 = 0.005
    alpha3 = 0.005
    K = 5
    split_seed = 1#1,2,3,4,5 are used in all five datasets
    
    
    edgepath = ['soc-sign-bitcoinalpha.csv','soc-sign-bitcoinotc.csv','wiki-RfA.txt',
                'soc-sign-Slashdot090221.txt','soc-sign-epinions.txt']
    count = np.loadtxt('input/count.txt').astype(int)
    datapath = edgepath[data_idx]
    
    
    train_features,train_labels,test_features,test_labels = \
        feature_and_label(data_idx,datapath,split_seed,count,K,alpha1,alpha2,alpha3)
    f1_micro,f1,f1_macro,auc = mlp(train_features,train_labels,test_features,test_labels)
    
    