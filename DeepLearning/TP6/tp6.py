import logging
logging.basicConfig(level=logging.INFO)

import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import click
from sklearn.model_selection import train_test_split
from datamaestro import prepare_dataset



from sklearn.datasets import fetch_openml

# Changer le DATA_PATH
DATA_PATH = "/home/lilou/Documents/M2/Deep_Learning/TP/data/MNIST/raw"
#"/tmp/mnist"

writer = SummaryWriter()

# Ratio du jeu de train à utiliser
TRAIN_RATIO = 0.05
TEST_RATIO = 0.2
VAL_RATIO   = 0.20   # 20% des 5% utilisés comme validation

BATCH_SIZE = 300

# chargemebt 

ds = prepare_dataset("com.lecun.mnist")

train_img, train_labels = ds.train.images.data(), ds.train.labels.data()
test_img,  test_labels  = ds.test.images.data(),  ds.test.labels.data()

n_train = int(TRAIN_RATIO * len(train_img))      # 3000 images
perm = np.random.permutation(len(train_img))[:n_train]

train_img_small = train_img[perm]
train_labels_small = train_labels[perm]

# normalisation 

train_img_small = train_img_small.astype(np.float32) / 255.0
test_img        = test_img.astype(np.float32)        / 255.0

# reshape PyTorch (Nb image, canaux, H, W)
train_img_small = train_img_small.reshape(-1, 1, 28, 28)
test_img        = test_img.reshape(-1, 1, 28, 28)


# split

X_train, X_val, y_train, y_val = train_test_split(
    train_img_small,
    train_labels_small,
    test_size=0.2,
    stratify=train_labels_small,
    random_state=0,
)


# objet numpy a convertir en torch pour creer des tensordataset

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train, dtype=torch.long)

X_val = torch.tensor(X_val)
y_val = torch.tensor(y_val, dtype=torch.long)

X_test = torch.tensor(test_img)
y_test = torch.tensor(test_labels, dtype=torch.long)


# creation dataset et dataloader

train_dataset = TensorDataset(X_train, y_train)
val_dataset   = TensorDataset(X_val,   y_val)
test_dataset  = TensorDataset(X_test,  y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False)

print(f"Train size: {len(train_dataset)}")
print(f"Val size:   {len(val_dataset)}")
print(f"Test size:  {len(test_dataset)}")

print("taille des entrée X_train : ", X_train.shape)   # torch.Size([2240, 1, 28, 28])
print("taille des y_train: ", y_train.shape)  # torch.Size([2240])


def entropy_from_logits(logits):
    """
    logits: taille (B, C)
    renvoie un tenseur (B,) avec l'entropie de chaque ehcnaillon
    """
    log_probs = F.log_softmax(logits, dim=1)   # (B, C)
    probs = torch.exp(log_probs)               # (B, C)
    entropy = -(probs * log_probs).sum(dim=1)      # (B,)
    return entropy

# =================== modèle et apprentissage ==================

EPOCHS = 1000

class Modele(nn.Module) :
    # reseau 3 couches liénaire avec 100 sorties + 1 couche de classification 10 sorties
    def __init__(self, input_size = 28*28, output_size = 10):
        super().__init__()

        # applique ne transfo sur la derniere dim et attend (batch, 28*28)
        self.fc1 = nn.Linear(input_size, 100)
        self.bn1 = nn.BatchNorm1d(100)

        self.fc2 = nn.Linear(100, 100)
        self.bn2 = nn.BatchNorm1d(100)

        self.fc3 = nn.Linear(100, 100)
        self.bn3 = nn.BatchNorm1d(100)

        self.fcout = nn.Linear(100, output_size)

        self.act = nn.ReLU()
        self.drop = nn.Dropout(p=0.2)
    
    def forward(self, x, retain_input_grads=False):
        # obliger de faire la discjonction sinon dans evaluate ca pose pb 
        #print("shape : ", x.shape)  # [300,784]

        x_in1 = x
        if retain_input_grads:
            x_in1.retain_grad()

        x = (self.act(self.bn1(self.fc1(x))))
        x_in2 = x
        if retain_input_grads:
            x_in2.retain_grad()

        x = self.drop(self.act(self.bn2(self.fc2(x))))
        x_in3 = x
        if retain_input_grads:
            x_in3.retain_grad()

        x = self.drop(self.act(self.bn3(self.fc3(x))))
        x_in4 = x
        if retain_input_grads:
            x_in4.retain_grad()

        x = self.drop(self.fcout(x))

        if retain_input_grads:
            return x, (x_in1, x_in2, x_in3, x_in4)
        else:
            return x, None
        

# boucle apprentissage 

#device = torch.device('cuda')
modele = Modele()
lr = 0.002
opti = torch.optim.Adam(modele.parameters(), lr = lr)
criterion = nn.CrossEntropyLoss()
l1_lambda = 1e-1  # penalisation 


def evaluate(modele, loader) :

    #print("Evaluation....")
    modele.eval()

    avg_eval_loss = 0.0
    acc = 0.0
    total_loss = 0.0
    total_echantillon = 0.0
    correct = 0


    with torch.no_grad():

        for X_img, y_label in loader :

            X_img = X_img.reshape(-1, 28*28)
            X = X_img
                
            logits, _ = modele(X, False)
            loss = criterion(logits, y_label)

            pred = logits.argmax(1)

            correct += (pred == y_label).sum().item()
            B = X.size(0)

            total_loss += loss.item()*B
            total_echantillon += B

        avg_eval_loss = total_loss/total_echantillon
        acc = correct/total_echantillon

    return avg_eval_loss, acc

def boucle_complete():

    #print("Entrainement...")
    


    for epoch in range(1, EPOCHS+1) :
        modele.train()
        # moyenne par epochs !!
        avg_train_loss = 0.0
        total_loss = 0.0
        total_echantillon = 0
        entropies_batches = []  

        for X_img, y_label in train_loader :
            
            X_img = X_img.reshape(-1, 28*28)
            X = X_img.clone().detach().requires_grad_(True)

            opti.zero_grad()

            
            logits, inputgrad = modele(X, True)
            #print(f"X_img : {X_img.shape}, y_label : {y_label.shape}, logits : {logits.shape}")
            loss = criterion(logits, y_label)

            # ajout normalisation L1 (on peut utilsier weight decay à regarder comment ca marche)
            l1_norm = sum(p.abs().sum() for p in modele.parameters())
            l2_norm = sum(p.pow(2).sum() for p in modele.parameters())

            loss = loss + l1_lambda*l2_norm + l1_lambda*l1_norm
        
            loss.backward()

            opti.step()

            B = X.size(0)  # batch_size
            total_echantillon += B 
            total_loss += loss.item()*B


            # stockage de l'entropy et calcule
            ent_batch = entropy_from_logits(logits).detach()
            entropies_batches.append(ent_batch)

        avg_train_loss = total_loss/total_echantillon
        avg_val_loss, acc_val = evaluate(modele, val_loader)
        entropies_epoch = torch.cat(entropies_batches)


        if epoch % 10 == 0 or epoch == 1:
            test_loss, test_acc = evaluate(modele, test_loader)
        else:
            test_loss, test_acc = float('nan'), float('nan')

       
        #print(f"epoch : {epoch}/{EPOCHS} |  train_loss : {avg_train_loss:.4f}  | acc_loss_val : {acc_val:.4f}  | val_loss {avg_val_loss:.4f} |  test acc = {test_acc:.4f} | test loss : {test_loss:.4f} ")
        print(f"[{epoch}/{EPOCHS}] "
              f"train_loss={avg_train_loss:.4f} |"
              f"val_loss={avg_val_loss:.4f} val_acc={acc_val:.4f} |"
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")

        writer.add_scalar("loss/train", avg_train_loss, epoch)
        writer.add_scalar("loss/val",   avg_val_loss,   epoch)
        writer.add_scalar("acc/val",    acc_val,        epoch)

        if not np.isnan(test_loss):
            writer.add_scalar("loss/test", test_loss, epoch)
            writer.add_scalar("acc/test",  test_acc,  epoch)

        if epoch % 40 == 0 or epoch == 1:
            writer.add_histogram("grads/in_fc1", inputgrad[0].grad, epoch)
            writer.add_histogram("grads/in_fc2", inputgrad[1].grad, epoch)
            writer.add_histogram("grads/in_fc3", inputgrad[2].grad, epoch)
            writer.add_histogram("grads/in_fcout", inputgrad[3].grad, epoch)

            writer.add_histogram("weights/fc1", modele.fc1.weight, epoch)
            writer.add_histogram("weights/fc2", modele.fc2.weight, epoch)
            writer.add_histogram("weights/fc3", modele.fc3.weight, epoch)
            writer.add_histogram("weights/fcout", modele.fcout.weight, epoch)

            writer.add_histogram("entropy/output_train", entropies_epoch, epoch)



# sans regularisation :
# surapprentissage enorme -> la loss train chute à 0 très rapidement 
# Courbe de validation remonte autour de 40 epochs


boucle_complete()

"""
dropout avec p = 0.5 -> ne suffit pas tjrs bcp d'overfitting et juste courbe très bruitée mtn 
ajout norme l2 et batchnorm > 

"""














"""
# chargement des données
mnist = fetch_openml("mnist_784", version=1, as_frame=False)

#print("type : ", type(mnist))
#print("info : " , mnist.frame.info())

X = mnist["data"].astype(np.float32)           # shape (70000, 784)
y = mnist["target"].astype(np.int64)

X /= 255.0

# reshape en (N, 1, 28, 28) pour MLP
X = X.reshape(-1, 1, 28, 28)

# split 
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X,
    y,
    test_size=TEST_RATIO,
    stratify=y,
    random_state=0,
)

X_small, _, y_small, _ = train_test_split(
    X_trainval,
    y_trainval,
    train_size=TRAIN_RATIO,
    stratify=y_trainval,
    random_state=0,
)

# 
# split de ces 5% en train et validation

X_train, X_val, y_train, y_val = train_test_split(
    X_small,
    y_small,
    test_size=VAL_RATIO,     
    stratify=y_small,
    random_state=0,
)
"""