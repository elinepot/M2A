import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.f_x = nn.Linear(input_size, hidden_size, bias=False) 
        self.f_h = nn.Linear(hidden_size, hidden_size)
        self.f_d = nn.Linear(hidden_size, output_size)


    def one_step(self, x, h):
        ''' x : un batch des sequences de taille batch*dim
        h: batch des états cachés de taille batch*latent
        sortie de taille batch*latent
        '''
        f = self.f_x(x) + self.f_h(h)
        h_new = torch.tanh(f)
        return torch.tensor(h_new, requires_grad=True)
    

    def forward(self, x, h):
        ''' 
        x : taille length*batch*input_dim
        h: taille batch*hidden_size
        sortie= : taille length*batch*hidden_size
        '''
        # /!\ dans exo2 : x de taille batch_size*length*input_dim
        # donc on fait x = torch.permtute(x, (2, 0, 1))
        
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
        # print('h', h.size())
        # print('self.f_d(h)', self.f_d(h).size())
        return self.f_d(h)


class SampleMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station

class ForecastMetroDataset(Dataset):
    def __init__(self, data,length=20,stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length= data,length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self,i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length-1)],self.data[day,(timeslot+1):(timeslot+self.length)]

