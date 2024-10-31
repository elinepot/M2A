import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import numpy as np
from datamaestro import prepare_dataset
from datetime import datetime

# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05

def store_grad(var):
    """Stores the gradient during backward

    For a tensor x, call `store_grad(x)`
    before `loss.backward`. The gradient will be available
    as `x.grad`

    """
    def hook(grad):
        var.grad = grad
    var.register_hook(hook)
    return var


####################################################################################################
# PREPARATION DONNEES
####################################################################################################

class MNISTDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
    

ds = prepare_dataset("com.lecun.mnist")
train_img, train_labels = ds.train.images.data(), ds.train.labels.data()
test_img, test_labels = ds.test.images.data(), ds.test.labels.data()

# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05
TRAIN_SIZE = int(TRAIN_RATIO * train_img.shape[0])
indices = np.random.choice(len(test_img), TRAIN_SIZE, replace=False)
subsampled_train_img = train_img[indices]
subsampled_train_labels = train_labels[indices]

# Create dataset objects
train_dataset = MNISTDataset(subsampled_train_img, subsampled_train_labels)
test_dataset = MNISTDataset(test_img, test_labels)

# Create dataloaders
BATCH_SIZE = 64
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



####################################################################################################
# MODELE
####################################################################################################

class Model(nn.Module):
    def __init__(self, INPUT_DIM, NUM_CLASSES, HIDDEN_SIZE, p_dropout, normalisation = "identity"):
        super().__init__()

        self.INPUT_DIM = INPUT_DIM
        self.NUM_CLASSES = NUM_CLASSES
        self.HIDDEN_LAYER = HIDDEN_LAYER
        self.p_dropout = p_dropout
        self.normalisation = normalisation


        self.layers = [nn.Linear(self.INPUT_DIM, self.HIDDEN_LAYER),
                nn.ReLU()]
        
        if self.normalisation == 'batchnorm':
            self.layers.append(nn.BatchNorm1d(self.HIDDEN_LAYER))

        if self.normalisation == 'layernorm':
            self.layers.append(nn.LayerNorm(self.HIDDEN_LAYER))
        
        self.layers.extend([nn.Linear(self.HIDDEN_LAYER, self.HIDDEN_LAYER),
                    nn.ReLU(),
                    nn.Linear(self.HIDDEN_LAYER, self.NUM_CLASSES)])
        
        if self.p_dropout is not None:
            # assert 0<=self.p_dropout or self.p_dropout<=1 , "p_dropout doit être une proba"
            self.layers.append(nn.Dropout(self.p_dropout))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)
    
    def train_mode(self):
        self.train()

    def test_mode(self):
        self.eval()
    

####################################################################################################
# RUN
####################################################################################################       

softmax = torch.nn.Softmax(dim=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = f'runs/tp7_{timestamp}'
writer = SummaryWriter(log_dir=log_dir)


def run(NB_EPOCHS, model, l1=False, l2=False):
    '''
    NB_EPOCHS : nombre d'epochs
    model : modèle\\
    l1 : hyperparametre de la regularisation L1 \\
    l2 : hyperparametre de la regularisation L2 \\
    '''

    # pour tensorboard
    title = f"lam1:{l1}_" if l1 is not False else ""
    title = f"lam2:{l2}_" if l2 is not False else ""
    title += f"p_dropout:{model.p_dropout}_" if model.p_dropout is not None else ""
    title += model.normalisation
    print(title)

    # enregistrer le gradient
    for name, param in model.named_parameters():
        param.retain_grad=True
    
    model.to(device)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
    loss_function = torch.nn.CrossEntropyLoss() 

    # assert not (l1 is not False and l2 is not False), "choisir une régularisation l1 ou l2"
    l = max(l1,l2)
    regularisation = 1 if l==l1 else 2


    # BOUCLE APPRENTISSAGE
    for epoch in tqdm(range(NB_EPOCHS)):

        loss_train = 0
        acc_train = 0
        entropy_train = 0
        loss_test = 0
        acc_test = 0
        best_acc_test=0
        entropy_test = 0

        # TRAIN
        model.train_mode()
        for x, y in train_loader:
            optimizer.zero_grad()
            x = (x.flatten(1).float()/255)
            x = torch.as_tensor(x).to(device)
            y = torch.as_tensor(y).to(device)
            y_hat = model(x)
        
            # regularisation
            all_linear_params = torch.cat([x.view(-1) for x in model.parameters()])
            norm = l*torch.norm(all_linear_params, regularisation)**regularisation
            
            loss = loss_function(y_hat, y) + norm
            loss.backward()
            optimizer.step()
            loss_train+=(loss.item())
            y_hat_softmax = softmax(y_hat)
            entropy_batch = -torch.sum(y_hat_softmax * torch.log(y_hat_softmax + 1e-9), dim=1).mean().item()
            entropy_train += entropy_batch
            acc = sum(y_hat_softmax.argmax(1)==y)/len(y)
            acc_train += acc
        
        avg_loss_train = loss_train/len(train_loader)
        avg_acc_train = acc_train/len(train_loader)

        writer.add_scalar('Train/Loss/'+title, avg_loss_train, epoch)
        writer.add_scalar('Train/Accuracy/'+title + title, avg_acc_train, epoch)

        
        # TEST
        model.test_mode()
        for x, y in test_loader:
            x = (x.flatten(1).float()/255)
            x = torch.as_tensor(x).to(device)
            y = torch.as_tensor(y).to(device)
            y_hat = model(x)
            y_hat_softmax = softmax(y_hat)
            entropy_batch = -torch.sum(y_hat_softmax * torch.log(y_hat_softmax + 1e-9), dim=1).mean().item()
            entropy_test += entropy_batch
            loss = loss_function(y_hat, y)
            loss_test+=loss.item()
            acc = sum(y_hat_softmax.argmax(1)==y)/len(y)
            acc_test += acc
        
        avg_loss_test = loss_test/len(test_loader)
        avg_acc_test = acc_test/len(test_loader)
        if best_acc_test<avg_acc_test:
            best_acc_test=avg_acc_test

        writer.add_scalar('Test/Loss/'+title, avg_loss_test, epoch)
        writer.add_scalar('Test/Accuracy/'+title, avg_acc_test, epoch)

        if epoch%int(NB_EPOCHS/20)==0: # une vingtaine seulement
            # on enregistre les poids et les biais des couches linéaires
            writer.add_histogram('Test/Entropy/'+title, entropy_test, epoch)
            writer.add_histogram('Train/Entropy/'+title, entropy_train, epoch)
            for name, param in model.named_parameters():
                writer.add_histogram(name+'/'+title, param, epoch) 
                writer.add_histogram(name+'/gradient/'+title, param.grad, epoch)

    return best_acc_test




INPUT_DIM = 28*28  # x.size[1]*x.size[2]
HIDDEN_LAYER = 25
NUM_CLASSES = 10
NB_EPOCHS = 50

model = Model(INPUT_DIM, NUM_CLASSES, HIDDEN_LAYER, p_dropout=None, normalisation="batchnorm")
run(NB_EPOCHS, model)
