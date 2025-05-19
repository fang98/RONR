# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
import pandas as pd



def read_graph(edge_path):
    if 'soc-sign-bitcoin' in edge_path:
        dataset = pd.read_csv(edge_path).values.tolist()
        G_tmp = nx.DiGraph()
        for i in range(len(dataset)):
            G_tmp.add_edge(dataset[i][0],dataset[i][1], weight=dataset[i][2])
    elif 'wiki-RfA' in edge_path:
        with open(edge_path, "r",encoding='UTF-8') as f:
            data = f.read()
        strlist = data.split('\n\n')
        del strlist[len(strlist)-1]
        G_tmp = nx.DiGraph()
        for i,str_i in enumerate(strlist):
            ebunch = str_i.split('\n')
            s_node = ebunch[0][4:]
            e_node = ebunch[1][4:]
            if ebunch[2][4:]=='-1':
                G_tmp.add_edge(s_node,e_node,weight = -1)
            elif ebunch[2][4:]=='1':
                G_tmp.add_edge(s_node,e_node,weight = 1)
    else:
        dataset = pd.read_csv(edge_path,sep='\t').values.tolist()
        G_tmp = nx.DiGraph()
        for i in range(len(dataset)):
            G_tmp.add_edge(dataset[i][0],dataset[i][1], weight=dataset[i][2])
    
    nodes = list(G_tmp.nodes())
    dict_nodes = {val:key for key,val in enumerate(nodes)}
    edges = {}
    edges["positive_edges"] = []
    edges["negative_edges"] = []
    G = nx.DiGraph()
    G.add_nodes_from(list(range(len(nodes))))
    for u,v in G_tmp.edges():
        if G_tmp[u][v]['weight']>0:
            G.add_edge(dict_nodes[u], dict_nodes[v], weight = 1)
            edges["positive_edges"].append([dict_nodes[u],dict_nodes[v]])
        else:
            G.add_edge(dict_nodes[u], dict_nodes[v], weight = -1)
            edges["negative_edges"].append([dict_nodes[u],dict_nodes[v]])
    edges["ecount"] = G.number_of_edges()
    edges["ncount"] = G.number_of_nodes()
    edges["graph"] = G

    return edges



edgepath = ['soc-sign-bitcoinalpha.csv','soc-sign-bitcoinotc.csv','wiki-RfA.txt',
            'soc-sign-Slashdot090221.txt','soc-sign-epinions.txt']
# edgepath = ['soc-sign-bitcoinalpha.csv']
n = 5
test_size = 0.2
seed = [i for i in range(1,n+1)]
count = []
for data in edgepath:
    datapath = 'data/'+data
    edges = read_graph(datapath)
    count.append([edges['ncount'],edges['ecount'],len(edges["positive_edges"]),len(edges["negative_edges"]),len(edges["positive_edges"])/len(edges["negative_edges"])])
    for i in range(n):
        print('dataset :',data,'; loop num :',str(i+1))
        split_seed = seed[i]
        positive_edges, test_positive_edges = train_test_split(edges["positive_edges"],
                                                               test_size=test_size,
                                                               random_state=split_seed)
        negative_edges, test_negative_edges = train_test_split(edges["negative_edges"],
                                                               test_size=test_size,
                                                               random_state=split_seed)
        
        positive_edges = np.array(positive_edges)
        test_positive_edges = np.array(test_positive_edges)
        negative_edges = np.array(negative_edges)
        test_negative_edges = np.array(test_negative_edges)
        
        positive_edges = np.hstack([positive_edges,np.ones([positive_edges.shape[0],1])])
        negative_edges = np.hstack([negative_edges,-1*np.ones([negative_edges.shape[0],1])])
        train_edges = np.vstack([positive_edges,negative_edges]).astype(int)
        
        test_positive_edges = np.hstack([test_positive_edges,np.ones([test_positive_edges.shape[0],1])])
        test_negative_edges = np.hstack([test_negative_edges,-1*np.ones([test_negative_edges.shape[0],1])])
        test_edges = np.vstack([test_positive_edges,test_negative_edges]).astype(int)
        
        np.save('input/'+data[:-4]+'_train_'+str(split_seed)+'.npy',train_edges)
        np.save('input/'+data[:-4]+'_test_'+str(split_seed)+'.npy',test_edges)


f = open('input/'+'count'+'.txt', 'w').close()
f = open('input/'+'count'+'.txt','a')
for j in range(len(count)):
    line = str(count[j][0])+' '+str(count[j][1])+' '+str(count[j][2])+\
        ' '+str(count[j][3])+' '+str(count[j][4])+'\n'
    f.writelines(line)
f.close()
