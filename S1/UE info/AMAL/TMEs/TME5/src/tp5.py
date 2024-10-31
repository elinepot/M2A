
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *

#  TODO: 

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.
    loss = torch.nn.CrossEntropyLoss(reduction='none')
    mask = output != padcar
    return loss(output*mask, target).mean()



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


####################################################################################
# RNN
####################################################################################

class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self, input_size, embedding_dim, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.emb = nn.Embedding(input_size, embedding_dim)
        self.f_x = nn.Linear(embedding_dim, hidden_size, bias=False) 
        self.f_h = nn.Linear(hidden_size, hidden_size)
        self.f_d = nn.Linear(hidden_size, output_size)


    def one_step(self, x, h):
        ''' 
        x : un batch des sequences de taille batch*dim
        h: batch des états cachés de taille batch*latent
        sortie de taille batch*latent
        '''
        print('x',x.size())
        x = self.emb(x)
        print('x_emb',x.size())
        f = self.f_x(x) + self.f_h(h)
        h_new = torch.tanh(f)
        return torch.tensor(h_new, requires_grad=True)
    

    def forward(self, x, h):
        ''' 
        x : taille length*batch*input_dim
        h: taille batch*hidden_size
        sortie= : taille length*batch*hidden_size
        '''
        # length, batch_size, input_dim = x.size()
        length, batch_size = x.size()
        _, hidden_size = h.size()  

        output = torch.zeros(length,batch_size,hidden_size)

        for i in range(length):
            output[i] = self.one_step(x[i, :], h)
            h = output[i].to(device)

        return output
    
    def decode(self, h):
        '''
        h: taille batch*latent
        sortie de taille batch*output
        '''     
        return self.f_d(h)


####################################################################################
# LSTM
####################################################################################


class LSTM(RNN):
    #  TODO:  Implémenter un LSTM
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)

        self.f = nn.Linear(input_size + hidden_size, hidden_size)
        self.i = nn.Linear(input_size + hidden_size, hidden_size)
        self.c = nn.Linear(input_size + hidden_size, hidden_size)
        self.o = nn.Linear(input_size + hidden_size, hidden_size)


    def one_step(self, x, h, C):
        concat_vec = torch.cat([h, x], dim=1) 
        
        f_t = torch.sigmoid(self.f(concat_vec))  # porte oubli
        i_t = torch.sigmoid(self.i(concat_vec)) # porte entrée
        C_t = f_t*C + i_t*torch.tanh(self.c(concat_vec))  # mise à jour (mémoire interne)
        o_t = torch.sigmoid(self.o(concat_vec)) # porte sortie
        h_t = o_t*torch.tanh(C_t)     # sortie
        return torch.tensor(h_t, requires_grad=True)
    
    def forward(self, x, h):
        length, batch_size, input_dim = x.size()
        _, hidden_size = h.size()  

        output = torch.zeros(length,batch_size,hidden_size)

        for i in range(length):
            output[i] = self.one_step(x[i, :, :], h)
            h = output[i].to(device)

        return output
    
    def decode(self, h):
        return super().decode(h)


####################################################################################
# GRU
####################################################################################

class GRU(RNN):
    #  TODO:  Implémenter un GRU
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__(input_size, hidden_size, output_size)

        self.f_z =nn.Linear(input_size + hidden_size, hidden_size)
        self.f_r= nn.Linear(input_size + hidden_size, hidden_size)
        self.f = nn.Linear(hidden_size, hidden_size)
    
    def one_step(self, x, h):
        concat_vec = torch.cat([h, x], dim=1) 
        z_t = torch.sigmoid(self.f_z(concat_vec))
        r_t = torch.sigmoid(self.f_r(concat_vec))
        concac_vec_bis = torch.cat(r_t*h,x)
        h_t = (1-z_t)*h + z_t*torch.tanh(self.f(concac_vec_bis))
        return torch.tensor(h_t, requires_grad=True)

    def forward(self, x, h):
        ''' 
        x : taille length*batch*input_dim
        h: taille batch*hidden_size
        sortie= : taille length*batch*hidden_size
        '''
        length, batch_size, input_dim = x.size()
        _, hidden_size = h.size()  

        output = torch.zeros(length,batch_size,hidden_size)

        for i in range(length):
            output[i] = self.one_step(x[i, :, :], h)
            h = output[i].to(device)

        return output
    
    def decode(self, h):
        '''
        h: taille batch*latent
        sortie de taille batch*output
        '''
        # return torch.nn.functional.softmax(self.f_d(h), dim=1)
        
        return self.f_d(h)


#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot
