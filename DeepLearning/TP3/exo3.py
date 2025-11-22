from utils import RNN, device,  ForecastMetroDataset

from torch.utils.data import  DataLoader
import torch
from torch.utils.tensorboard import SummaryWriter
import datetime

# Nombre de stations utilisé
CLASSES = 10
#Longueur des séquences
LENGTH = 20
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 1
#Taille du batch
BATCH_SIZE = 32

DIM_LATENT = 64

PATH = "/home/lilou/Documents/M2/Deep_Learning/TP/TP3/TP3bis/data/"


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"))
ds_train = ForecastMetroDataset(
    matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = ForecastMetroDataset(
    matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)

#  TODO:  Question 3 : Prédiction de séries 
# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runsq3"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

modele = RNN(DIM_LATENT, DIM_INPUT, DIM_INPUT)   
eps = 0.001
# lr = 0.01 trop grand ici 
opti = torch.optim.Adam(modele.parameters(), lr = eps)
criterion = torch.nn.MSELoss()

# ========== Apprentissage =================
Epochs = 15

print("Entrainement pour préduire le prochain flux ")
for epoch in range(Epochs):

    modele.train()
    total_loss = 0.0
    total_samples = 0

   # print("epoch : ",epoch)

    for batchX, batchY in data_train :
       # print("entrainement")
        opti.zero_grad()

        # batchX: [L, B, C, D]  → [L, B*C, D]
        L, B, C, D = batchX.shape
        batchX = batchX.view(L, B * C, D)
        batchY = batchY.view(L, B * C, D)
        h0 = torch.zeros(batchX.size(1), DIM_LATENT)
        h = modele(batchX, h0) 
        # h = [h0, h1, ..., h_T-1, h_T]  et mtn on decode selon tous les etats cachés à chaque fois pas que le dernier
        # voir config photo many to many
       
        # prédiction
        y_hat = modele.decode(h)

        # calcule de l'erreur, loss
        loss = criterion(y_hat, batchY)

        #backprop
        loss.backward()

        opti.step()

        # on veut calculer les errerus moyenne sur tous les batch pas que le dernier, donc on accumule et on 
        # affiche pour chaque epoch

        

        batch_size = batchX.size(1)

        total_samples += batch_size
        total_loss += loss.item() * batchX.size(1)

    epoch_loss = total_loss / total_samples
    rmse = torch.sqrt(torch.tensor(epoch_loss))
    print(f"Epoch {epoch+1}/{Epochs} |       Train RMSE: {rmse:.4f}")

    writer.add_scalar("Rmse/Train", rmse, epoch)



    modele.eval()
    total_loss_test = 0.0
    total_samples = 0
    
    with torch.no_grad() :
        for batchX, batchY in data_test :
          #  print("evaluation")
            # batchX: [L, B, C, D]  → [L, B*C, D]
            L, B, C, D = batchX.shape
            batchX = batchX.view(L, B * C, D)
            batchY = batchY.view(L, B * C, D)
            h0 = torch.zeros(batchX.size(1), DIM_LATENT)
            h = modele(batchX, h0) 
                 
            # prédiction
            y_hat = modele.decode(h)

            # calcule de l'erreur, loss
            loss_test = criterion(y_hat, batchY)

            batch_size = batchX.size(1)
            total_samples += batch_size
            total_loss_test += loss_test.item() * batchX.size(1)

    epoch_loss = total_loss_test / total_samples
    rmse_test = torch.sqrt(torch.tensor(epoch_loss))
    print(f"Epoch {epoch+1}/{Epochs} | Test RMSE: {rmse_test:.4f}")

    writer.add_scalar("Rmse/Train", rmse_test, epoch)
