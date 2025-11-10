from pathlib import Path
import os
import torch
import random
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
from tp1 import MSE, Linear, Context
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from datamaestro import prepare_dataset
from pathlib import Path
ds = prepare_dataset("com.lecun.mnist");
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
# ========= fixer la graine -> rendre les calculs reproductibles ==========

def set_seed(seed=42):
    # Fixe la graine Python
    random.seed(seed)

    # Fixe la graine NumPy
    np.random.seed(seed)

    # Fixe la graine PyTorch (CPU)
    torch.manual_seed(seed)

    # Fixe la graine PyTorch (GPU)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Pour rendre les calculs déterministes (reproductibles)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# ======= Question 1 : descente de gradient avec torch.optim =========

# initialisation 

X = torch.randn(50, 13, dtype=torch.float32)
Y = torch.randn(50, 3, dtype=torch.float32)  # Y vrai à approcher
eps = 0.05
nb_epoch = 150


# poids du modèle
W = torch.nn.Parameter(torch.randn(13,3))
b = torch.nn.Parameter(torch.randn(3))

def scratch_opti(X,W,b,Y,eps) :
    """descente sans torch.opti"""

    train_loss = []

    for i in range(nb_epoch):

        # y_hat = X@W+b
        Y_hat = Linear.apply(X,W,b)
        loss = MSE.apply(Y_hat, Y)
        # accumule les gradients 
        loss.backward()

        writer.add_scalar('Loss/train scratch', loss.item(), i)

        # descente de gradient
        with torch.no_grad():
            W -= eps * W.grad
            b -= eps * b.grad

        W.grad.zero_()
        b.grad.zero_()

        train_loss.append(loss.item)
        
        #if (i + 1) % 10 == 0:
        print(f"Epoch {i+1:3d} | Loss = {loss.item():.6f}")
    
    print("\nEntraînement avec from scratch terminé.")
        


def SGD_opti(X,W,b,Y, eps) :
    """descente SGD avec torch.opti"""
    # on optimise selon w et b 
    train_loss = []

    optim = torch.optim.SGD(params = [W,b], lr = eps)
    optim.zero_grad()   # initialisation des gradients (mis à 0)

   # print(f"paramètres init : W = {W}, b = {b}")

    # boucle d'entrainement

    for i in range(nb_epoch):

        # y_hat = X@W+b
        Y_hat = Linear.apply(X,W,b)
        loss = MSE.apply(Y_hat, Y)
        loss.backward()
        # accumule les gradients

        # tensorboard --logdir runs/
        writer.add_scalar('Loss/train SGD', loss.item(), i)

        #if i%50 == 0 :
            # fais somme sur les 50 gradients (somme des 100 gradients -> mini batch)
         #   optim.step()
          #  optim.zero_grad()

        optim.step()
        optim.zero_grad()

        train_loss.append(loss.item)
        
        #if (i + 1) % 10 == 0:
        print(f"Epoch {i+1:3d} | Loss = {loss.item():.6f}")
    
    print("\nEntraînement avec SGD terminé.")
  #  print(f"paramètres finaux : W = {W}, b = {b}")



def Adam_opti(X,W,b,Y,eps) :
    # on optimise selon w et b 
    train_loss = []

    optim = torch.optim.Adam(params = [W,b], lr = eps)
    optim.zero_grad()   # initialisation des gradients (mis à 0)

    # boucle d'entrainement
    for i in range(nb_epoch) :

        Y_hat = Linear.apply(X,W,b)
        loss = MSE.apply(Y_hat, Y)
        # accumule les gradients
        loss.backward()
    
        # tensorboard --logdir runs/
        writer.add_scalar('Loss/train ADAM', loss.item(), i) 

        # mise à jour 
        optim.step()
        optim.zero_grad()

        print(f"Epoch {i+1:3d} | Loss = {loss.item():.6f}")

        train_loss.append(loss.item())

    print("\nEntraînement avec Adam terminé.")


#SGD_opti(X,W,b,Y,eps)
#Adam_opti(X,W,b,Y,eps)
#scratch_opti(X,W,b,Y,eps)

# interpretation : Adam plutot bruité, SGD non. Atteignent environ la meme valeur les deux 


# ========== Question 2 - modules ===========

input_dim = 13
output_dim = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X = torch.randn(50, input_dim, dtype=torch.float32)
X = X.to(device)
Y = torch.randn(50, output_dim, dtype=torch.float32)  # Y vrai à approcher
Y = Y.to(device)

# poids du modèle -> on a pas besoin de les expliciter car quand dans un reseau on utilise nn.Linear
# python sait qu'on a besoin d'une matrice de poids et vecteur de biais pour les paramètres d'une couche linéaire

# implémentation d'un réseau à deux couches
class Modele_un(torch.nn.Module):
    def __init__(self, input_dim = input_dim, output_dim = output_dim, hidden_dims = [100]) :
        super(Modele_un, self).__init__()    # appel au constructeur de la mère 
        
        self.fc1 = torch.nn.Linear(input_dim, hidden_dims[0])
        self.fc2 = torch.nn.Linear(hidden_dims[0], output_dim)

        #act = torch.nn.Tanh() # -> c'est une classe 

    def forward(self, x):
        # appel a la fonction Tanh
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)

        return x


# ========= definir liste de paramètres du réseau ===========

# entré du réseau : meme dimension que l'entrée du réseau (2e coord de x = dim)
#x = torch.randn(100, input_dim)

# implémentatoon de la boucle de descentre avec torch.optim



def train_MLP_un(X, W, b, Y, nb_epoch = 100, eps = 0.01) :

    # definition de l'optimiseur
    model = Modele_un()
    model = model.to(device)    # chargement des paramètres sur device
    criterion = torch.nn.MSELoss()
    # model.parameters
    opti = torch.optim.Adam(params = model.parameters(), lr=eps)
    # cmt il connait cest qui les paramètres, avant dans la q°1 on les avait explicité ?
    opti.zero_grad()

    model.train()

    if model and criterion and opti:
        print("Model Architecture:")
        print(model)
        #print("Paramètres : ", model.parameters())
        print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters())}")
        print(f"\nLoss function: {criterion}")
        print(f"Optimizer: {opti.__class__.__name__}")

    for i in range(nb_epoch):

        # forward pass 
        Y_hat = model(X)
        loss = criterion(Y_hat, Y)
        # backward pass 
        loss.backward()

        writer.add_scalar("Loss/train q2", loss.item(), i)

        # mise à jour
        opti.step()
        opti.zero_grad()

        print(f"itération {i+1:3d} | Loss = {loss.item():.6f}")
    
    print("Entrainement MLP1 terminé")

#train_MLP_un(X, W, b, Y, nb_epoch = 100, eps = 0.01)
# MLP très efficace, rapide et loss atteint minimum mieux que les autres modèees

"""
============  version avec sequential  ============ 
model = nn.Sequential(
    nn.Linear(10, 100),
    nn.Tanh(),
    nn.Linear(100, 1)
)

loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(500):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()
    if epoch % 50 == 0:
        print(f"[Sequential] Epoch {epoch}: loss = {loss.item():.4f}")


Différences : model Sequential utile si on a très peu de couche sinon utilsier une classe est plus flexible est réutilisable    

"""

# ========= question 3 data Loader ===============

DATA_PATH = "/home/lilou/Documents/M2/Deep_Learning/TP/TP2/src"
X,y = fetch_openml('mnist_784', return_X_y = True, as_frame = False, data_home = DATA_PATH)
#X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)



# séparation train set, test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# Fit et transofrm seulement sur le train
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)       


X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.astype(int), dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.astype(int), dtype=torch.long)

# implémenter un dataset -> implementer une classe qui héirite de dataset
# doit definitir deux méthodes __getitem__  -> conteneur[i] et __len__ -> longueur

class MonDataset(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return (self.x[index], self.y[index])
    
    def __len__(self):
        return len(self.x)
    

train_dataset = MonDataset(X_train_tensor, y_train_tensor)
test_dataset = MonDataset(X_test_tensor, y_test_tensor)

data_train = DataLoader(train_dataset, shuffle = True, batch_size = 128)
data_test = DataLoader(test_dataset, shuffle = True, batch_size = 128)

"""
test dataset 

i = 0
for batch_x, batch_y in data_train :
    print(f"taille : ", batch_x.shape, batch_y.shape)
    if i == 3 :
        break
    i += 1
"""

# ====== 5 - Checkpointing =========
# state_dict() renvoie sous la forme d'un dictionnaire les paramètres du modèle + paramtre de l'optimiseur

savepath = Path("model_lilou.pch")

class State :
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch = 0
        self.iteration = 0

"""
EXEMPLE CHECKPOINTING 

# si le fichier existe deja, on recharge l'etat complet 
if savepath.is_file() :
    with savepath.open("rb") as fp :
        state = torch.load(fp)
else :   # sinon on initialise un nouveau modèle et un nouvel optimiseur 
    model = Modele_un()
    optim = torch.optim.Adam(params = model.parameters(), lr=eps)
    state = State(model, optim)

ITERATIONS = 100
# si l’entraînement a ete interrompu  à epoch=5, on reprend à 5.
for epoch in range(state.epoch, ITERATIONS):     # on reprend à l'epoch sauvegardée
    for x, y in data_train:

        state.optim.zero_grad()
        x = x.to(device)
        xhat = state.model(x)
        l = torch.nn.MSELoss()(xhat, x)
        l.backward()
        state.optim.step()

        state.iteration += 1

    # on sauvegarde l etat a la fin de chaque epoch
    with savepath.open("wb") as fp:
        state.epoch = epoch + 1
        torch.save(state, fp)

"""


# ========= 6 - Implémentation d'un autoencodeur ==========

# dim_sortie = dim_entrée

input_dim = 28*28
latent_dim = 32   # compression de l'image dans un espace de dim 32

# encodeur :  entrée () -> linéaire -> relu -> latent 
# decodeur : latent() ->  sigmoide -> sortie ()

class Autoencodeur(nn.Module):
    def __init__(self, input_dim = input_dim, latent_dim = latent_dim):
        super(Autoencodeur, self).__init__()
        
        #encoder
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.relu = nn.ReLU()
        
        # decoder
        self.decoder = nn.Linear(latent_dim, input_dim)
        self.sigmoid = nn.Sigmoid()
        
        self.decoder.weight.data.copy_(self.encoder.weight.data.T.clone())
        
    def forward(self, x):
        z = self.encoder(x)     
        z = self.relu(z)         
        output = self.decoder(z) 
        output = self.sigmoid(output)  
        return output


# ======= entrainement question 5 ===========


# preparation du modele 
model = Autoencodeur().to(device)


def train_autoencoder(model, train_loader, test_loader, epochs=20, lr=0.001, check_path ="autoencoder.pch"):

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    writer = SummaryWriter()

    savepath = Path(check_path)
    start_epoch = 0

    # charger checkpoint si existe
    if savepath.is_file():
        with savepath.open("rb") as fp:
            state = torch.load(fp, map_location=device)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        start_epoch = state['epoch'] + 1
        print(f"Reprise à l’époque {start_epoch}")

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0

        for batchX, _ in train_loader : 

            # BUT : recosntruire x on cherche f tq xhat = f(x) donc y = xhat

            optimizer.zero_grad()
            batchX = batchX.view(-1, 28*28).to(device) 
            xhat = model(batchX)
            loss = criterion(xhat, batchX)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        

        with savepath.open("wb") as fp:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
                }, fp)

        # evaluer sur test
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for batchX, _ in test_loader:
                batchX = batchX.view(-1, 28*28).to(device)
                outputs = model(batchX)
                test_loss += criterion(outputs, batchX).item()

        test_loss /= len(test_loader)

        # TensorBoard (affichage)
        writer.add_scalar("Loss/train auto", train_loss, epoch)
        writer.add_scalar("Loss/test auto", test_loss, epoch)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f}")

    writer.close()


#model = Autoencodeur().to(device)
#train_autoencoder(model, data_train, data_test, epochs=20, lr=0.001)


# ============== Highway papier ================

# classique couche y=H(x) 
# highway couche = y = H(x)T(x) + H(x)(1-T(X))
# H(x) couche linéaire + activation
# T(X) porte de transofmration sigma(...)
# entrainement rapide indepednanmment de la profondeur -> pas de pb de vanishing/exploding gradient



class HighwayLayer(nn.Module):
    """
    y = H(x) * T(x) + x * (1 - T(x))
    avec H(x) = activation(W_H x + b_H), T(x) = sigmoid(W_T x + b_T).
    """
    def __init__(self, dim: int, activation=F.relu, gate_bias: float = -2.0):
        super().__init__()
        self.dim = dim
        self.H = nn.Linear(dim, dim, bias=True)   # transform
        self.T = nn.Linear(dim, dim, bias=True)   # porte
        self.activation = activation
        nn.init.constant_(self.T.bias, gate_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Hx = self.activation(self.H(x))
        Tx = torch.sigmoid(self.T(x))
        return Hx * Tx + x * (1.0 - Tx)


class HighwayNetwork(nn.Module):
    """

    """
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 num_layers: int,
                 output_dim: int | None = None,
                 activation=F.relu,
                 gate_bias: float = -2.0):
        super().__init__()
        self.proj = nn.Identity() if input_dim == hidden_dim else nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList([
            HighwayLayer(hidden_dim, activation=activation, gate_bias=gate_bias)
            for _ in range(num_layers)
        ])
        self.head = nn.Identity() if output_dim is None else nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


class HighwayAutoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=128, num_highway=4, latent_dim=32):
        super().__init__()
        # encodeur
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc2 = nn.Linear(hidden_dim, latent_dim)
        # bloc highway sur l'espace latent (dim fixe = latent_dim)
        self.highway = HighwayNetwork(input_dim=latent_dim, hidden_dim=latent_dim,
                                      num_layers=num_highway, output_dim=None,
                                      activation=F.relu, gate_bias=-2.0)
        # décodeur
        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # encode
        z = F.relu(self.enc1(x))
        z = F.relu(self.enc2(z))
        # highway au centre
        z = self.highway(z)
        # decode
        y = F.relu(self.dec1(z))
        y = torch.sigmoid(self.dec2(y))
        return y
    

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x = x.view(x.size(0), -1).to(device)  # MNIST aplati si images
        y = y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)







