from utils import random_walk,construct_graph
import math
from tqdm import tqdm
import networkx as nx
from torch import nn
from torch.utils.data import DataLoader
import random
import torch
from torch.utils.tensorboard import SummaryWriter

import time
import logging

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


##  TODO: 
## QUESTION 1 : 
## Ecrire une classe TripletDataset qui renvoie un triplet de nœuds ancre, positif, négatif 
# (ou le positif et le négatif sont tirés au hasard dans le voisinage et hors du voisinage de l’ancre


N = 100 # nombre de noeuds
D = 128 # dimension de l'espace latent
MARGIN = 1.0 # marge de la loss
P = 2 # The norm degree for pairwise distance
EPOCHS = 1e-7 # Small constant for numerical stability.



class TripletDataset(torch.utils.data.Dataset):
    def __init__(self, nodes2id, id2nodes, graph):
        self.nodes2id = nodes2id
        self.id2nodes = id2nodes
        self.graph = graph 

    def __len__(self):
        return len(self.nodes2id)

    def __getitem__(self, idx):
        neighbors = list(self.graph.neighbors(self.id2nodes[idx]))
    
        p = random.choice(neighbors)  # p dans le voisinage
        n = random.choice(list(set(self.id2nodes) - set(neighbors))) # n pas dans le voisinage
        a = idx

        return self.nodes2id[a], self.nodes2id[p], self.nodes2id[n]
        


## Implémenter la boucle d’apprentissage pour la triplet loss et apprendre une représentation des films. 
# Vous pouvez visualiser la représentation avec l’algorithme t-sne (en particulier dans tensorboard avec la fonction add_embedding).



if __name__=="__main__":
    PATH = "ml-latest-small/ml-latest-small/"
    logging.info("Constructing graph")
    movies_graph, movies = construct_graph(PATH + "movies.csv", PATH + "ratings.csv")
    logging.info("Sampling walks")
    walks = random_walk(movies_graph,5,10,1,1)
    nodes2id = dict(zip(movies_graph.nodes(),range(len(movies_graph.nodes()))))
    id2nodes = list(movies_graph.nodes())
    id2title = [movies[movies.movieId==idx].iloc[0].title for idx in id2nodes]
    ##  TODO: 
    N = len(movies_graph.nodes())
    
    dataset = TripletDataset(nodes2id, id2nodes, movies_graph)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
    logging.info("Training")

    LR = 0.001

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(10):
        for i, (a, p, n) in enumerate(dataloader):
    
            loss = triplet_loss(a, p, n)
            print("loss : ", loss)
            loss.backward()

            

    
