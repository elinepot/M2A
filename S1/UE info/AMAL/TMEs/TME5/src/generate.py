from textloader import  string2code, id2lettre
import math
import torch

#  TODO:  Ce fichier contient les différentes fonction de génération

def generate(rnn, emb, decoder, eos, start="", maxlen=200):
    """  Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). 
        Initialise le réseau avec start (ou à 0 si start est vide) et 
        génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """

    #  TODO:  Implémentez la génération à partir du RNN, et d'une fonction decoder qui renvoie les logits (logarithme de probabilité à une constante près, i.e. ce qui vient avant le softmax) des différentes sorties possibles
    
    # model = rnn
    # input_size = model.input_size
    # hidden_size = model.hidden_size
    # output_size = model.output_size

    x = start
    len = 1
    sentence = x

    # while x!= eos & len<maxlen:
    #     x_embeded = emb(x)
    #     h= torch.zeros((x.shape[0], hidden_size))
    #     output_sequence = rnn.forward(x, h)
    #     print(output_sequence.size())
    #     last_hidden_state = output_sequence[-1, :, :]
    #     x_next = rnn.decode(last_hidden_state).argmax(1)
    #     sentence += x_next
    #     x = x_next

    while x!=eos and len<maxlen:
        x_embed = emb(x)
        x_next = decoder(rnn(x_embed)).argmax(1)
        sentence+=x_next
        x = x_next
    
    return sentence


def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200):
    """
        Génere une séquence en beam-search : 
        à chaque itération, on explore pour chaque candidat les k symboles les plus probables; 
        puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés 
        (au sens de la vraisemblance) pour l'itération suivante.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * k : le paramètre du beam-search
        * start : début de la phrase
        * maxlen : longueur maximale
    """
    #  TODO:  Implémentez le beam Search


# p_nucleus
def p_nucleus(decoder, alpha: float):
    """Renvoie une fonction qui calcule la distribution de probabilité sur les sorties

    Args:
        * decoder: renvoie les logits étant donné l'état du RNN
        * alpha (float): masse de probabilité à couvrir
    """
    def compute(h):
        """Calcule la distribution de probabilité sur les sorties

        Args:
           * h (torch.Tensor): L'état à décoder
        """
        #  TODO:  Implémentez le Nucleus sampling ici (pour un état s)
    return compute
