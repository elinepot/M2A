import torch
from torch import nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm


#==============================================================================
## Import des données
#==============================================================================

writer = SummaryWriter()

data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()

x = torch.tensor(datax,dtype=torch.float)
y = torch.tensor(datay,dtype=torch.float).reshape(-1,1)


#==============================================================================
## II. Différenciation automatique : autograd
#==============================================================================


#### Régression linéaire

class Linear(nn.Module):
    @staticmethod
    def forward( x, w, b): 
        yhat = x@w + b
        return yhat 
           

#### Descente de gradient par mini batch

w = torch.randn(x.shape[1], y.shape[1])
b = torch.randn(y.shape[1])
epsilon = 0.0005
w.requires_grad=True
b.requires_grad=True
num_epochs = 100
batch_size = 10
linear = Linear()
loss = nn.MSELoss()

for epoch in range(num_epochs):
    for batch_start in range(0, x.shape[0], batch_size):
        # batch de données
        x_batch = x[batch_start:batch_start+batch_size]
        y_batch = y[batch_start:batch_start+batch_size]
        
        yhat = linear.forward(x_batch, w, b)

        l = loss(yhat, y_batch)
        print(f"Epoch {epoch}, batch = {batch_start}: loss {l}")
        writer.add_scalar('Loss/train', l, epoch)
        
        l.backward() 
        
        with torch.no_grad():
            w -= epsilon * w.grad
            b -= epsilon * b.grad
        # réinitialisation
        w.grad.data.zero_() 
        b.grad.data.zero_() 

print("done !")


#### Descente de gradient stochastique

w = torch.randn(13, 3)
b = torch.randn(3)
epsilon = 0.0005
w.requires_grad=True
b.requires_grad=True
num_epochs = 100
linear = Linear()
loss = nn.MSELoss()

for epoch in range(num_epochs):
    for i in range(x.shape[0]):
        # Sélectionne un exemple de données aléatoire
        random_index = torch.randint(0, x.shape[0], (1,))
        x_sample = x[random_index]
        y_sample = y[random_index]
        
        yhat = linear.forward(x_sample, w, b)
    
        l = loss(yhat, y_sample)
        print(f"Epoch {epoch}, SGD Step {i}: loss {l.item()}") 
        writer.add_scalar('Loss/train', l, epoch)

        l.backward()

        with torch.no_grad():  
            w -= epsilon * w.grad
            b -= epsilon * b.grad
        # réinitialisation des gradients
        w.grad.zero_()
        b.grad.zero_()

print("done !")


#==============================================================================
## IV. Module
#==============================================================================

# Implémenter un réseau à 2 couches : lineaire → tanh → lineaire → MSE

#### Sans conteneur :

class Reseau(nn.Module):
    def __init__(self, x, y, hidden_size):
        super(Reseau, self).__init__()
        self.x = x
        self.y = y
        self.hidden_size = hidden_size
        self.input_size = x.shape[0]
        self.output_size = y.shape[0]
        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)
    
    def forward(self, x):
        assert x.shape[1] == self.input_size
        yhat = self.linear2(self.tanh(self.linear1(x)))
        return yhat


# Descente de gradient SGD:
hidden_size = 3
epsilon = 0.0005
num_epochs = 25
learning_rate = 0.0005
linear = Linear()
loss = nn.MSELoss()
model = Reseau(x[0], y[0], hidden_size)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i in range(x.shape[0]):
        random_index = torch.randint(0, x.shape[0], (1,))
        x_sample = x[random_index]
        x_sample.reshape(1, -1)
        y_sample = y[random_index]
        
        yhat = model.forward(x_sample)

        l = loss(yhat, y_sample)
        print(f"Epoch {epoch}, SGD Step {i}: loss {l.item()}") 
        writer.add_scalar('Loss/train', l, epoch)
    
        optimizer.step()
        optimizer.zero_grad()

print("done !")


#### Avec conteneur :

class Reseau_conteneur(nn.Module):
    def __init__(self, x, y, hidden_size):
        super(Reseau_conteneur, self).__init__()
        self.x = x
        self.y = y
        self.hidden_size = hidden_size
        self.input_size = x.shape[0]
        self.output_size = y.shape[0]

        self.reseau = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, self.output_size)
        )
 
    def forward(self, x):
        assert x.shape[1] == self.input_size
        yhat = self.reseau(x)
        return yhat


# Descente de gradient SGD:
hidden_size = 3
epsilon = 0.0005
num_epochs = 25
learning_rate = 0.0005
loss = nn.MSELoss()
model = Reseau_conteneur(x[0], y[0], hidden_size)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for i in range(x.shape[0]):
        random_index = torch.randint(0, x.shape[0], (1,))
        x_sample = x[random_index]
        x_sample.reshape(1, -1)
        y_sample = y[random_index]
        
        yhat = model.forward(x_sample)

        l = loss(yhat, y_sample)
        print(f"Epoch {epoch}, SGD Step {i}: loss {l.item()}") 
        writer.add_scalar('Loss/train', l, epoch)
    
        optimizer.step()
        optimizer.zero_grad()

print("done !")
