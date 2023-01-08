
import networkx as nx
import torch
import dgl
 
N=6
DAMP=0.85
K=10
 
g=dgl.DGLGraph()
g.add_nodes(6)
 

g.add_edges([1,1,1,2,2,4,0,5,5],[2,3,4,3,5,3,1,3,4])

g.ndata["pagerank"]=torch.ones(N)/N
g.ndata["degree"]=g.out_degrees(g.nodes()).float()
 
def pagerankMessageFunc(edges):
    return {"pv" :edges.src["pagerank"]/edges.src["degree"]}
 
def pagerankReduceFunc(nodes):
    msgs=torch.sum(nodes.mailbox["pv"],dim=1)
    pv=(1-DAMP)/N+msgs*DAMP
    return {"pv":pv}
 
g.register_message_func(pagerankMessageFunc)
g.register_reduce_func(pagerankReduceFunc)
 
def  pagerankIteration(g):
    for u,v in zip(*g.edges()):
        g.send((u,v))
    for v in g.nodes():
        g.recv(v)
 
 
pagerankIteration(g)
print(g.ndata["pagerank"])