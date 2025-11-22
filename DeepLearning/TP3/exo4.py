import string
import unicodedata
import torch
import sys
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset,DataLoader

from utils import RNN, device

BATCH_SIZE = 32
## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

PATH = "/home/lilou/Documents/M2/Deep_Learning/TP/TP3/TP3bis/data/"

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
         # (sequence x, séquence cible (y))  (la meme seq decalee d un cran)



#  TODO: 

data_trump = DataLoader(TrumpDataset(open(PATH+"trump_full_speech.txt","rb").read().decode(),maxlen=1000), batch_size= BATCH_SIZE, shuffle=True)

# Taille du vocabulaire (nombre de symboles = nombre de caractères)
vocab_size = len(LETTRES) + 1    
dim_input = 64                   # dimension de l'embedding
dim_latent = 128                 # dimension de l'etat cache
dim_output = vocab_size          # dimension sortie = nombre de symboles considere



model = RNN(dim_latent, dim_input, dim_output)  
eps = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr = eps)
criterion = torch.nn.CrossEntropyLoss()

epochs = 15

# --- Boucle d'apprentissage --- TRES TRES LONG
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_chars = 0

    for x, y in data_trump:
        x, y = x.to(device), y.to(device)
        print("vu")
        x_onehot = torch.nn.functional.one_hot(x, num_classes=vocab_size).float()

        # Projection linéaire : [batch, length, dim_input]
        x_embed = model.embedding_layer(x_onehot)

        # reformatage : [length, batch, dim_input]
        x_embed = x_embed.permute(1, 0, 2)
        
        h0 = torch.zeros(x_embed.size(1), dim_latent).to(device)

        # pass dans RNN
        H = model.forward(x_embed, h0)  # [length, batch, dim_latent]
        Y_hat = model.decode(H)         # [length, batch, vocab_size]

        # reshape pour CrossEntropy : [length * batch, vocab_size] (il attend ( (B,C), (C))
        loss = criterion(Y_hat.view(-1, vocab_size), y.T.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy caractère par caractère
        preds = Y_hat.argmax(dim=2)           # [length, batch]
        total_correct += (preds.T == y).sum().item()
        total_chars += y.numel()

    acc = total_correct / total_chars
    print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(data_trump):.4f} | Accuracy: {acc*100:.2f}%")



