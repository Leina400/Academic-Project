from utils import RNN, device,SampleMetroDataset
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import datetime

# Nombre de stations utilisé
CLASSES = 10  # nbre de classe à predire
#Longueur des séquences (t= 0 à t = 19)
LENGTH = 20   
# Dimension de l'entrée (1 (in) ou 2 (in/out))
DIM_INPUT = 1
#Taille du batch
BATCH_SIZE = 32  # 32 jours 
# dimension état caché 
DIM_LATENT = 64

PATH = PATH = "/home/lilou/Documents/M2/Deep_Learning/TP/TP3/TP3bis/data/"

# Tensorboard : rappel, lancer dans une console tensorboard --logdir runs
writer = SummaryWriter("runs/runs"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))


matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch","rb"))
ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length = LENGTH, stations_max = ds_train.stations_max)
data_train = DataLoader(ds_train,batch_size=BATCH_SIZE,shuffle=True)
data_test = DataLoader(ds_test, batch_size=BATCH_SIZE,shuffle=False)


#  TODO:  Question 2 : prédiction de la ville correspondant à une séquence

# Objectif : à partir d'une séquence x_t, on veut savoir à quelle stations apaprtient cette séquence
# contexte many to one : softmax + cross entropy 
# décodage indique la classe de la séquence 

#writer.add_scalar('Loss/train scratch', loss.item(), i)

# init et préparation du modèle et de ses paramètres :

modele = RNN(DIM_LATENT, DIM_INPUT, CLASSES)   # [0,0,1,0,0,0,0,0,0,0] exemple de sortie
eps = 0.001
# lr = 0.01 trop grand ici 
opti = torch.optim.Adam(modele.parameters(), lr = eps)
criterion = torch.nn.CrossEntropyLoss()

# device = 

Epochs = 15

for epoch in range(Epochs):

    modele.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

   # print("epoch : ",epoch)

    for batchX, batchY in data_train :
       # print("entrainement")
        opti.zero_grad()

        # forward pass
        batchX = batchX.permute(1, 0, 2)
        h0 = torch.zeros(batchX.size(1), DIM_LATENT)
        H = modele(batchX, h0) 
        h_last = H[-1]    # H = [h0, h1, ..., h_T-1, h_T]  et nous on veut h_T, le dernier qui contient toute l'info necessaire
       
        # pr"diction
        y_hat = modele.decode(h_last)

        # calcule de l'erreur, loss
        loss = criterion(y_hat, batchY)

        #backprop
        loss.backward()

        opti.step()

        # on veut calculer les errerus moyenne sur tous les batch pas que le dernier, donc on accumule et on 
        # affiche pour chaque epoch

        

        batch_size = batchX.size(1)
        total_loss += loss.item() * batch_size  # somme des pertes totales (on fait 1/B * B)
        preds = torch.argmax(y_hat, dim=1)      # classes prédites
        total_correct += (preds == batchY).sum().item()
        total_samples += batch_size

    # moyenne sur chaque époque 
    epoch_loss = total_loss / total_samples
    epoch_acc = total_correct / total_samples

    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.add_scalar("Accuracy/train", epoch_acc, epoch)

    print(f"Epoch : {epoch+1}/{Epochs} |      Train Loss: {epoch_loss:.4f} | Train Accuracy: {epoch_acc:.4f}")


    modele.eval()
    total_loss_test = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad() :
        for batchX, batchY in data_test :
          #  print("evaluation")
            # forward pass
            batchX = batchX.permute(1, 0, 2)
            h0 = torch.zeros(batchX.size(1), DIM_LATENT)
            H = modele(batchX, h0) 
            h_last = H[-1]    # H = [h0, h1, ..., h_T-1, h_T]  et nous on veut h_T, le dernier qui contient toute l'info necessaire
        
            # prédiction
            y_hat = modele.decode(h_last)

            # calcule de l'erreur, loss
            loss_test = criterion(y_hat, batchY)

            # accuracy 
            preds = torch.argmax(y_hat, dim = 1)
            total_correct += (preds == batchY).sum().item()

            # loss
            batch_size = batchX.size(0)
            total_loss_test += loss_test.item() * batch_size 

            total_samples += batch_size

    # moyenne sur chaque époque 
    epoch_loss_test = total_loss_test / total_samples
    epoch_acc_test = total_correct / total_samples

    writer.add_scalar("Loss/test", epoch_loss_test, epoch)
    writer.add_scalar("Accuracy/test", epoch_acc_test, epoch)

    print(f"Epoch : {epoch+1}/{Epochs} | Test Loss: {epoch_loss_test:.4f} | Test Accuracy: {epoch_acc_test:.4f}")


#INTERPRETATION :
# train_acc autour de 50% pas ouf, semble underfitting mais test_acc autour de 70% donc generalise bien
# underfit -> mon modele est simpliste

