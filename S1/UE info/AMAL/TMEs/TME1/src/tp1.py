# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/piwowarski/venv/amal/3.7/bin/activate

import torch
from torch.autograd import Function
from torch.autograd import gradcheck

import numpy as np

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
        q = y.size()[0]
        mse = (1/q)*torch.norm(yhat - y)**2
        return mse
        
    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)
        q = y.size()[0]
        dy = (-2/q)*(yhat - y)
        dyhat = (2/q)*(yhat - y)
    
        return dyhat*grad_output, dy*grad_output
    
mse = MSE.apply

#  TODO:  Implémenter la fonction Linear(X, W, b)

class Linear(Function):
    """Début d'implementation de la fonction Linear"""
    @staticmethod
    def forward(ctx, x, w, b):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(x, w, b)

        #  TODO:  Renvoyer la valeur de la fonction
        yhat = x@w + b
        return yhat
        
    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        x, w, b = ctx.saved_tensors
        #  TODO:  Renvoyer par les deux dérivées partielles (par rapport à yhat et à y)
        dx = w
        dw = x.T
        db = grad_output.sum(0)
        
        return grad_output@dx.T, dw@grad_output, db
    
linear = Linear.apply