
import torch
from torch.autograd import Function
from torch.autograd import gradcheck


#print("test")

class Context:
    """Un objet contexte très simplifié pour simuler PyTorch

    Un contexte différent doit être utilisé à chaque forward
    """
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):
    """Début d'implementation de la fonction MSE"""
    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)

        #  TODO:  Renvoyer la valeur de la fonction
        return ((yhat - y) ** 2).mean() 

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        # grad_output -> gradient qui arrive de la couche suivante
        yhat, y = ctx.saved_tensors
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)
        # Gradients

        dL_dyhat = grad_output * 2 * (yhat - y)
        dL_dy = grad_output * -2 * (yhat - y)

        return dL_dyhat, dL_dy

#  TODO:  Implémenter la fonction Linear(X, W, b)sur le même modèle que MSE

class Linear(Function):
    """ implémentation de la fonction lineaire """

    @staticmethod
    def forward(ctx, X, W, b):
        ctx.save_for_backward(X,W,b)

        return X@W+b
    
    @staticmethod
    def backward(ctx, grad_output):
        X, W, b = ctx.saved_tensors

        # verif les dim

        grad_X = grad_output @ W.T
        grad_W = X.T @ grad_output
        grad_b = grad_output.sum(dim=0, keepdim=True)         # pas bonne dim la ??


        return grad_X, grad_W, grad_b
    


## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply

