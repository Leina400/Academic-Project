import torch
import torch.nn as nn
from torch.utils.data import Dataset
import string

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
vocab_size = len(LETTRES) + 1  


class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1

    def __init__(self, dim_latent, dim_input, dim_output):
        super(RNN, self).__init__() 

        # Transformation entree x_t vers l’espace latent
        self.Wi = nn.Linear(dim_input, dim_latent)

        # Transformation de l'etat caché precedant h_{t-1}
        self.Wh = nn.Linear(dim_latent, dim_latent)

        # Décodage de l’état caché vers la sortie
        self.Wd = nn.Linear(dim_latent, dim_output)

        # exo4 : Projection one-hot -> embedding continu dans espace des entrees 
        self.embedding_layer = nn.Linear(vocab_size, dim_input, bias=False)

        self.activation1 = nn.Tanh()
        self.activation2 = nn.Softmax()

    def one_step(self,x,h):
        """
        x : [batch, dim]
        h = h_t-1 : [batch, latent]
        retour : [batch, latent]
        correspond a l'etape h_t = f(x_t, h_t-1)
        """

        h_t = self.activation1(self.Wi(x)+self.Wh(h))
        return h_t
    
    def forward(self, x, h0):
        """
        x : [lenght, batch, dim]
        h : [batch, latent]
        retour : sequence des etats cachés [lenght, batch, latent]
        """
        length, _, _ = x.shape
        h = h0
        H = []

        #print("Shape entrée dans forward:", x.shape)

        for i in range(length):

          #  print("x[i] shape =", x[i].shape)
           # print("h shape   =", h.shape)
            h = self.one_step(x[i], h)    # forme : [batch, dim_latent]
            H.append(h.unsqueeze(0))      # cree une dimension en 0 qu'on remplira apres
                                          # forme: [1, batch, dim_latent]

        H = torch.cat(H, dim = 0)         # cree la dimension length en premiere coordonée

        return H



    def decode(self, h):
        """
        correspond à y = d(h_t)
        """
        y_t = self.activation2(self.Wd(h))
        #y_t = self.Wd(h)
        return y_t


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

