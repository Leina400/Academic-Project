import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import MSE, Linear, Context
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader,TensorDataset, Dataset


# Les données supervisées
x = torch.randn(50, 13,  requires_grad=True, dtype=torch.float64)
y = torch.randn(50, 3,  requires_grad=True, dtype=torch.float64)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3,  requires_grad=True, dtype=torch.float64)
b = torch.randn(1, 3, requires_grad=True, dtype=torch.float64)

epsilon = 0.05

writer = SummaryWriter()
for n_iter in range(100):
    ##  TODO:  Calcul du forward (loss)


    yhat = Linear.apply(x, w, b)    #xw+b  -> execute  Linear.forward(x, w, b)
    loss = MSE.apply(yhat, y)

    # `loss` doit correspondre au coût MSE calculé à cette itération
    # on peut visualiser avec
    # tensorboard --logdir runs/ 
    writer.add_scalar('Loss/train', loss, n_iter)

    # Sortie directe
   # print(f"Itérations {n_iter}: loss {loss}")

    ##  TODO:  Calcul du backward (grad_w, grad_b)

    #loss.backward() 
    grads = torch.autograd.grad(loss, (w, b))
    grad_w, grad_b = grads

    ##  TODO:  Mise à jour des paramètres du modèle

    with torch.no_grad():
        w -= epsilon * grad_w  #W.grad
        b -= epsilon * grad_b  # b.grad

    # W.grad.zero_()

    #print(f"n_iter {n_iter+1}: loss = {loss.item():.4f}")



writer.close()

## Chargement des données Boston et transformation en tensor.

data = fetch_california_housing(data_home="./data/") ## chargement des données
data_x = torch.tensor(data['data'],dtype=torch.float) 
data_y = torch.tensor(data['target'],dtype=torch.float) 

X_train, X_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.33, random_state=42)

# A NORMALISER

#print("Shape : ", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# Convertir en tenseurs
X_train = torch.tensor(X_train, dtype=torch.float64)
Y_train = torch.tensor(y_train, dtype=torch.float64)
X_test = torch.tensor(X_test, dtype=torch.float64)
Y_test = torch.tensor(y_test, dtype=torch.float64)

# Paramètres du modèle
n_features = X_train.shape[1]  # 8
n_outputs = 1
W = torch.randn(n_features, n_outputs, dtype=torch.float64, requires_grad=True)
b = torch.randn(1, n_outputs, dtype=torch.float64, requires_grad=True)

n_epochs = 150 
eps = 1e-4  

writer = SummaryWriter()

for epoch in range(n_epochs):

    # forward pass
    yhat_train = Linear.apply(X_train, W, b)
    loss_train = MSE.apply(yhat_train, Y_train)

    # calcul des gradient de la loss
    loss_train.backward()

    # mise a jour des paramètres W et b
    with torch.no_grad():
        W -= epsilon * W.grad
        b -= epsilon * b.grad

    # Remise à zéro des gradients
    W.grad.zero_()
    b.grad.zero_()

    # forward sur le test (sans backward, on evalue juste)
    with torch.no_grad():
        yhat_test = Linear.apply(X_test, W, b)
        loss_test = MSE.apply(yhat_test, Y_test)

    # === Logs ===
    writer.add_scalar("Loss/train", loss_train.item(), epoch)
    writer.add_scalar("Loss/test", loss_test.item(), epoch)

    print(f"Epoch {epoch+1}: Train Loss = {loss_train.item():.4f}, Test Loss = {loss_test.item():.4f}")

writer.close()



