from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime

import matplotlib.pyplot as plt

#==============================================================================
## Téléchargement des données
#==============================================================================

from datamaestro import prepare_dataset
ds = prepare_dataset("com.lecun.mnist")
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels =  ds.test.images.data(), ds.test.labels.data()

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1,3,1,1).double()/255.
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f'samples', images, 0)


savepath = Path("model.pch")

#==============================================================================
## I. Gérer les données avec Dataset et DataLoader
#==============================================================================

from torch.utils.data import Dataset, DataLoader

class MonDataset(Dataset):
    def __init__(self, data, labels ) :
        self.data = data
        self.labels = labels   

    def __getitem__(self, index) :
        return self.data[index], self.labels[index]

    def __len__(self ) :
        return len(self.data)

# Creation du dataloader , en spécifiant la taille des batchs, et l'ordre aléatoire
BATCH_SIZE = 32
data_train = DataLoader (MonDataset (train_images, train_labels), shuffle=True, batch_size=BATCH_SIZE)
data_test = DataLoader (MonDataset (test_images, test_labels), shuffle=True, batch_size=BATCH_SIZE)



#==============================================================================
## II. Implémentation d'un autoencodeur
#==============================================================================

## On a codé et testé deux autoencodeurs selon des méthodes différentes:

#### Première version:

# Téléchargement des données

from datamaestro import prepare_dataset

ds = prepare_dataset("com.lecun.mnist")
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels = ds.test.images.data(), ds.test.labels.data()

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

# Pour visualiser
# Les images doivent etre en format Channel (3) x Hauteur x Largeur
images = (
    torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1, 3, 1, 1).double() / 255.0
)
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f"samples", images, 0)

savepath = Path("model.pch")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
images.to(device)


class State:
    def _init_(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch = 0
        self.iteration = 0

    def save(self, path):
        torch.save(self, path)


class AutoEncoder(nn.Module):
    def _init_(
        self,
        x,
        n_code: int,
        encoder=None,
        decoder=None,
        weights=None,
        bias_encode=None,
        bias_decode=None,
    ):
        super()._init_()
        try:
            self.n_x = x.shape[0]
        except:
            self.n_x = len(x)
        self.n_latent = n_code
        if weights is None:
            self.weights = nn.Parameter(torch.Tensor(self.n_latent, self.n_x)).double()
        else:
            self.weights = weights
        if bias_encode is None:
            self.bias_encode = nn.Parameter(torch.Tensor(self.n_latent)).double()
        else:
            self.bias_decode = bias_decode
        if bias_decode is None:
            self.bias_decode = nn.Parameter(torch.Tensor(self.n_x)).double()
        else:
            self.bias_decode = bias_decode
        if encoder is None:
            self.encoder = nn.Sequential(
                F.linear(x, self.weights, bias=self.bias_encode), nn.ReLU(True)
            )
        else:
            self.encoder = encoder
        self.encoder(x)
        if decoder is None:
            self.decoder = nn.Sequential(
                F.linear(self.encoder(x), self.weights.T, self.bias.decode),
                nn.Sigmoid(torch.Tensor(self.n_x)),
            )
        else:
            self.decoder = decoder
        self.decoder(x)

    def forward(self, x):
        latent = self.encoder(x)
        y = self.decoder(latent)
        return y, latent


LATENT_SPACE_SIZE = 100
LEARNING_RATE = 0.05
ITERATIONS = 10000
if savepath.is_file():
    state = State(model, optim)
    state = torch.load(savepath)
else:
    model = AutoEncoder(images.flatten(-1), LATENT_SPACE_SIZE)
    model.to_device(device)
    optim = torch.optim.Adam(model.parameters, LEARNING_RATE)
    state = State(model, optim)
for epoch in range(state.epoch, ITERATIONS):
    for x, y in train_loader:
        state.optim.zero_grad()
        x.to_device(device)
        x_hat = state.model(x)
        l = loss(xhat, x)
        l.backward()
        state.optim.step()
        state.iteration += 1
    with savepath.open("wb") as file:
        state.epoch += 1
        state.save(file)

output = model(images)
output.to("cpu")
print(output)

#### Deuxième version:

class Autoencodeur(nn.Module):
    def __init__(self, n_x, n_x_compress, weight=None, biais_encoder=None, biais_decoder=None):
        super(Autoencodeur, self).__init__()
        self.n_x = n_x
        self.n_x_compress = n_x_compress

        if weight is None:
            self.weight = nn.Parameter(torch.randn(self.n_x, self.n_x_compress).double())
        else:
            self.weight = nn.Parameter(weight.double())

        if biais_encoder is None:
            self.biais_encoder = nn.Parameter(torch.randn(self.n_x_compress).double())
        else:
            self.biais_encoder = nn.Parameter(biais_encoder.double())

        if biais_decoder is None:
            self.biais_decoder = nn.Parameter(torch.randn(self.n_x).double())
        else:
            self.biais_decoder = nn.Parameter(biais_decoder.double())

    def encode(self, x):
        x = x.to(self.weight.dtype)
        x = x.reshape(1, x.shape[0])
        return torch.relu(F.linear(x, self.weight.t(), self.biais_encoder))

    def decode(self, x_compress):
        return torch.sigmoid(F.linear(x_compress, self.weight, self.biais_decoder))

    def forward(self, x):
        x_compress = self.encode(x)
        x_compress_decompress = self.decode(x_compress)
        return x_compress, x_compress_decompress
    

images = (
    torch.tensor(train_images[0:8]).unsqueeze(1).repeat(1, 3, 1, 1).double() / 255.0
)
# Permet de fabriquer une grille d'images
images = make_grid(images)
# Affichage avec tensorboard
writer.add_image(f"samples", images, 0)

model = Autoencodeur(images.shape[0], 10)
x_compress, x_compress_decompress = model.forward(images)
print(x_compress)
      

#### Test sur une image et un label:
def petit_test():
    images, labels = next(iter(data_train))
    image, label = images[0], labels[0]
    w, h = image.shape

    plt.imshow(image) 
    plt.show()

    image = image.flatten()
    model = Autoencodeur(image.shape[0], 10)
    x_compress, x_compress_decompress = model.forward(image)
    image_compress_decompress = x_compress_decompress[0]
    print("compression done!")

    image_compress_decompress = image_compress_decompress.detach()
    image_compress_decompress = image_compress_decompress.reshape(w, h)

    plt.imshow(image_compress_decompress)
    plt.show()

#petit_test()


#==============================================================================
## IV. Checkpointing
#==============================================================================

from pathlib import Path

class State:
    def _init_(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch = 0
        self.iteration = 0

    def save(self, path):
        torch.save(self, path)



##### Expériences pour l’autoencodeur

def run_autoencodeur():
    savepath = Path("autoencodeur.pch")
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')

    writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    n_x_compress = 32
    alpha =  1e-3

    if savepath.is_file():
        with savepath.open ("rb") as fp :
            state = torch.load (fp)
    else:
        autoencoder = Autoencodeur(len(train_images[0].flatten()), n_x_compress)
        autoencoder = autoencoder.to(device)
        optim = torch.optim.SGD(params = autoencoder.parameters(), lr = alpha)
        state = State(autoencoder, optim)


    ITERATIONS = 100

    loss = nn.MSELoss()
    losses_train = []
    losses_test = []

    train_loader = DataLoader (MonDataset (train_images, train_labels), shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader (MonDataset (test_images, test_labels), shuffle=True, batch_size=BATCH_SIZE)

    for epoch in range (state.epoch, ITERATIONS) :

        # train 
        for x, y in train_loader :
            state.optim.zero_grad()
            x = x.to(device)
            xhat = [autoencoder.forward(pic.flatten())[1].reshape(x[0].shape) for pic in x]
            xhat = torch.stack(xhat)
            x = x.to(xhat.dtype)
            l = loss(xhat, x)
            losses_train.append(l)
            l.backward()
            state.optim.step()
            state.iteration += 1
        with savepath.open("wb") as fp :
            state.epoch = epoch + 1
            torch.save (state, fp)

        print("train done!")

        # test
        for x, y in test_loader :
            x = x.to(device)
            xhat = [autoencoder.forward(pic.flatten())[1].reshape(x[0].shape) for pic in x]
            xhat = torch.stack(xhat)
            x = x.to(xhat.dtype)
            l = loss(xhat, x)
            losses_test.append(l)
        print("test done!")

        print(f'epoch n° {epoch} : erreur de train: { np.mean(losses_train)}')

#run_autoencodeur()


##### Highway network

class Highway(nn.Module):
    def __init__(self, n_x, L):
        # L : nombre de couches
        super(Highway, self).__init__()
        self.n_x = n_x
        self.L = L
        self.WH = nn.Parameter(torch.randn(self.n_x, self.n_x).double())
        self.WT = nn.Parameter(torch.randn(self.n_x, self.n_x).double())
        self.WC = nn.Parameter(torch.randn(self.n_x, self.n_x).double())
        self.bH = nn.Parameter(torch.randn(self.n_x).double())
        self.bT = nn.Parameter(torch.randn(self.n_x).double())
        self.bC = nn.Parameter(torch.randn(self.n_x).double())

    def forward(self, x):
        x = x.to(self.WH.dtype)
        x = x.reshape(1, x.shape[0])
        for layer in range(self.L):
            H = torch.relu(F.linear(x, self.WH.t(), self.bH))
            T = torch.sigmoid(F.linear(x, self.WT.t(), self.bT))
            C = torch.sigmoid(F.linear(x, self.WC.t(), self.bC))
            x = H*T + x*C
        return x
    

def run_highway():
    savepath = Path("highway.pch")
    device = torch.device('cuda'if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    alpha =  1e-3
    L = 5

    if savepath.is_file():
        with savepath.open ("rb") as fp :
            state = torch.load (fp)
    else:
        highway = Highway(len(train_images[0].flatten()), L)
        highway = highway.to(device)
        optim = torch.optim.SGD(params = highway.parameters(), lr = alpha)
        state = State(highway, optim)

    ITERATIONS = 100

    loss = nn.MSELoss()
    losses_train = []
    losses_test = []

    train_loader = DataLoader (MonDataset (train_images, train_labels), shuffle=True, batch_size=BATCH_SIZE)
    test_loader = DataLoader (MonDataset (test_images, test_labels), shuffle=True, batch_size=BATCH_SIZE)

    for epoch in range (state.epoch, ITERATIONS) :

        # train 
        for x, y in train_loader :
            state.optim.zero_grad()
            x = x.to(device)
            xhat = [highway.forward(pic.flatten()).reshape(x[0].shape) for pic in x]
            xhat = torch.stack(xhat)
            x = x.to(xhat.dtype)
            l = loss(xhat, x)
            losses_train.append(l)
            l.backward()
            state.optim.step()
            state.iteration += 1
        with savepath.open("wb") as fp :
            state.epoch = epoch + 1
            torch.save (state, fp)

        print("train done!")

        # test
        for x, y in test_loader :
            x = x.to(device)
            xhat = [highway.forward(pic.flatten()).reshape(x[0].shape) for pic in x]
            xhat = torch.stack(xhat)
            x = x.to(xhat.dtype)
            l = loss(xhat, x)
            losses_test.append(l)
        print("test done!")

        print(f'epoch n° {epoch} : erreur de train: { np.mean(losses_train)}')

#run_highway()
