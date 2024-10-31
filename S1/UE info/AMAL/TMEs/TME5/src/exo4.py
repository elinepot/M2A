import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader

from tp5 import RNN, device, GRU, LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if  c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)


class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)
    
    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0),dtype=torch.long),t])
        return t[:-1],t[1:]



#  TODO: 
PATH = "C:/Users/eline/OneDrive/Bureau/M2A/AMAL/student_tp4-1/data/"

class MyEmbeddingLayer(nn.Module):
    def __init__(self, n_symboles, output_size):
        super().__init__()
        self.n_symboles = n_symboles
        self.output_size = output_size

        self.projection = torch.nn.Linear(self.n_symboles, self.output_size)

    def forward(self, x, label=False):
        x_one_hot =  torch.nn.functional.one_hot(x, self.n_symboles).float()
        if not label:
            x_proj = self.projection(x_one_hot)
            return x_proj
        return x_one_hot

##########################################################################################
# ENTRAINEMENT
##########################################################################################

BATCH_SIZE = 128
HIDDEN_SIZE = 100
DIM_OUTPUT = len(id2lettre)  # sortie du RNN = nombre de symboles considérés   # = 96

# prendre un RNN qui permet de engendrer un discours à la Trump :
data_trump = DataLoader(TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=100), batch_size= BATCH_SIZE, shuffle=True)


EMBED_DIM = 75 # < len(id2lettre = 96)
embedding = MyEmbeddingLayer(len(id2lettre), EMBED_DIM)
embedding.to(device)

model = RNN(input_size=len(id2lettre), embedding_dim=EMBED_DIM,hidden_size=HIDDEN_SIZE, output_size=DIM_OUTPUT)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
embed_optimizer = torch.optim.Adam(embedding.parameters(), lr=0.01)


loss_train_per_epoch = []

for epoch in range(10):
    print("\n__________Epoch :", epoch, "___________________________________")
    epoch_loss = 0
    for i, (x, y) in enumerate(data_trump):
        x = x.long().to(device)
        y = y.to(device)
        y =  torch.nn.functional.one_hot(y, num_classes=EMBED_DIM).float() 
        # x = torch.permute(x, (1,0))
        print(x.size())  # x de taille 100x128 ie de taille hidden_sizexbatch_size
        # y_projete = torch.permute(y_projete, (1,0,2))

        optimizer.zero_grad()
        embed_optimizer.zero_grad()

        h = torch.zeros((x.size(1), HIDDEN_SIZE), device=device)
        output_sequence = model.forward(x, h)
        last_hidden_state = output_sequence.to(device) # 50, 128, 100

        y_hat = model.decode(last_hidden_state)

        loss = criterion(y_hat, y)
        loss.backward()

        optimizer.step()
        embed_optimizer.step()
        
        epoch_loss += loss.item()
        loss_train_per_epoch.append(epoch_loss / len(data_trump))

        torch.cuda.empty_cache()

        if i % 50 ==0:
            print("Epoch %d, Batch %d, Loss %f" % (epoch, i,  loss.item()))

    
    print('Texte de Trump:', code2string(y[0]))

    print('Prediction:', code2string(y_hat.argmax(2)[:,0]))


##########################################################################################
# GENERATION
##########################################################################################
print("Generation de texte façon D. Trump :")

caractere = torch.tensor(torch.randint(len(lettre2id), (1,))).long().to(device)
texte = [caractere]

for i in range(100):
    h = torch.zeros((1, HIDDEN_SIZE), device=device)
    x = embedding(texte[-1])
    output = model.one_step(x, h)
    output = model.decode(output)
    caractere = output.argmax(1)
    texte.append(caractere)
texte = torch.cat(texte, dim=0)
texte = texte.tolist()
print(code2string(texte))