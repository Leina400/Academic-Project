
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from textloader import *
from generate import *
import datetime
from textloader import  string2code, id2lettre

#  TODO: 

DATA_PATH = "/home/lilou/Documents/M2/Deep_Learning/TP/TP3/TP3bis/data/"

writer = SummaryWriter("runs/runsTP4Trump_"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

def maskedCrossEntropy(output: torch.Tensor, target: torch.LongTensor, padcar: int):
    """
    :param output: Tenseur length x batch x output_dim,
    :param target: Tenseur length x batch
    :param padcar: index du caractere de padding
    """
    #  TODO:  Implémenter maskedCrossEntropy sans aucune boucle, la CrossEntropy qui ne prend pas en compte les caractères de padding.

    # PyTorch ne sait pas calculer directement une cross-entropie sur un tenseur 3D
    # on veut : output_flat a taille (N, vocab_size) et target_flat a taille (N,)
    output_flat = output.reshape(-1, output.size(-1))  # concatene (applati) toutes les dimensions sauf la denriere 
    target_flat = target.reshape(-1) # applati

    # Masque binaire : 1 si different de padcar
    mask = (target_flat != padcar).float()

    # cross entro elem par elem (sans réduction)
    loss_flat = torch.nn.functional.cross_entropy(output_flat, target_flat, reduction='none')

    # Moyenne pondere sur les positions valides
    loss = (loss_flat * mask).sum() / mask.sum()

    return loss




class RNN(nn.Module):
    #  TODO:  Implémenter comme décrit dans la question 1

    def __init__(self, dim_latent, dim_input, dim_output):
        super(RNN, self).__init__() 

        # entree xt vers l espace latent
        self.Wi = nn.Linear(dim_input, dim_latent)

        # projection de l'etat cache precedant ht-1
        self.Wh = nn.Linear(dim_latent, dim_latent)

        # decode etat cache vers la sortie
        self.Wd = nn.Linear(dim_latent, dim_output)

        # nn.embedding -> embedding continu dans espace des entrees (projection linéaire dans espace des embedding)
        #self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim_output)
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=dim_input)


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
        # entrée du prochain pas 
    
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
        #y_t = self.activation2(self.Wd(h))
        y_t = self.Wd(h)
        return y_t
    



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding = nn.Embedding(vocab_size, input_size)
        
        # matrice pour les differentes gate
        self.Wf = nn.Linear(input_size + hidden_size, hidden_size)  # Porte d'oubli
        self.Wi = nn.Linear(input_size + hidden_size, hidden_size)  # Porte d'entrée
        self.Wo = nn.Linear(input_size + hidden_size, hidden_size)  # Porte de sortie
        self.Wg = nn.Linear(input_size + hidden_size, hidden_size)  # Candidat pour l'état cellulaire
        
        # decodeur  : couche linéaire pour transformer l'état caché en logits
        self.decoder = nn.Linear(hidden_size, output_size)
        
    def one_step(self, x, h_prev, c_prev):
     
        combined = torch.cat([x, h_prev], dim=1)
        
        ft = torch.sigmoid(self.Wf(combined))  # Porte d'oubli
        it = torch.sigmoid(self.Wi(combined))  # Porte d'entrée
        ot = torch.sigmoid(self.Wo(combined))  # Porte de sortie
        
   
        gt = torch.tanh(self.Wg(combined))  # Candidat pour l'état cellulaire
        
        # Mettre à jour l'état cellulaire
        c_t = ft * c_prev + it * gt
        
        # Calculer l'état caché
        h_t = ot * torch.tanh(c_t)
        
        return h_t, c_t

    def forward(self, X, h0=None, c0=None):
        """
        X : Tenseur de taille (batch_size, seq_len) (indices des caractères ou mots)
        h0 : Tenseur de taille (batch_size, hidden_size) pour l'état caché initial
        c0 : Tenseur de taille (batch_size, hidden_size) pour l'état cellulaire initial
        """
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        
        # on convertit les indices en embeddings
        X = self.embedding(X)  # X est maintenant de taille (batch_size, seq_len, input_size)
        
        # on init des états cachés et cellulaires
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=X.device)
        if c0 is None:
            c0 = torch.zeros(batch_size, self.hidden_size, device=X.device)
        
        latent_states = []
        h, c = h0, c0
        
        # on calcule des états cachés pour chaque pas de temps
        for t in range(seq_len):
            h, c = self.one_step(X[:, t, :], h, c)
            latent_states.append(h)
        
        # on retourne les états cachés pour chaque pas de temps
        return torch.stack(latent_states, dim=1), (h, c)  # Shape: (batch_size, seq_len, hidden_size)
    
    def decode(self, h):
        """
        Transforme les états cachés en logits.
        
        :param h: Tenseur de taille (batch_size, seq_len, hidden_size)
        :return: Tenseur de taille (batch_size, seq_len, output_size) (les logits)
        """
        return self.decoder(h)

class GRU_(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, vocab_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.embedding = nn.Embedding(vocab_size, input_size)

        self.Wz = nn.Linear(input_size + hidden_size, hidden_size)  # zt : porte de mise à jour
        self.Wr = nn.Linear(input_size + hidden_size, hidden_size)  # rt : porte de réinitialisation
        self.W = nn.Linear(input_size + hidden_size, hidden_size)   # ht : état caché actuel
        
      
        self.decoder = nn.Linear(hidden_size, output_size)
        
    def one_step(self, x, h_prev):
        
        # S'assurer que x et h_prev sont sur le même device
        assert x.device == h_prev.device, "x et h_prev ne sont pas sur le même device"
        
        """
        Prend un seul pas de temps (x) et l'état caché précédent (h_prev).
        """
     
        combined = torch.cat([x, h_prev], dim=1)

        # pn calcule les portes zt et rt
        zt = torch.sigmoid(self.Wz(combined))  # Porte de mise à jour
        rt = torch.sigmoid(self.Wr(combined))  # Porte de réinitialisation

        combined_reset = torch.cat([x, rt * h_prev], dim=1)
        h_tilde = torch.tanh(self.W(combined_reset))  # Candidat pour l'état caché

        # etat cache
        ht = (1 - zt) * h_prev + zt * h_tilde

        return ht

    def forward(self, X, h0=None):
        """
        Prend en entrée un batch de séquences (X) et l'état caché initial (h0),
        renvoie les états cachés à chaque pas de temps.
        X : Tenseur de taille (batch_size, seq_len) (indices des caractères ou mots)
        h0 : Tenseur de taille (batch_size, hidden_size)
        """
        batch_size = X.shape[0]
        seq_len = X.shape[1]

        # Convertir les indices en embeddings
        X = self.embedding(X)  # X est maintenant de taille (batch_size, seq_len, input_size)
        
        # Initialisation de l'état caché h0
        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=X.device)
        
        latent_states = []
        h = h0
        
        # Calcul des états cachés pour chaque pas de temps
        for t in range(seq_len):
            h = self.one_step(X[:, t, :], h)  # Passage d'un pas de temps à l'autre
            latent_states.append(h)
        
        # Retourne les états cachés pour chaque pas de temps
        return torch.stack(latent_states, dim=1)  # Shape: (batch_size, seq_len, hidden_size)

    def decode(self, h):
        """
        Transforme les états cachés en logits.
        
        :param h: Tenseur de taille (batch_size, hidden_size)
        :return: Tenseur de taille (batch_size, output_size) (les logits)
        """
        return self.decoder(h)


# ========= entrainer un RNN sur prédire le prochain caractère ==========
# entrée : "Bonjour" caractère position t=0, ... , t=L-1
# target : "onjour"  caractère position t=1,     , t=L-1


vocab_size = len(id2lettre)
print(vocab_size)
dim_latent = 128
dim_input = 64
dim_output = vocab_size
batch_size = 32
lr = 1e-2
epochs = 60
batch_size = 64
device = "cuda" if torch.cuda.is_available() else "cpu"

ds = TextDataset(open(DATA_PATH+"trump_full_speech.txt","rb").read().decode(), maxlen=1000)
loader = DataLoader(ds, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

# on definit le modèle
modele = RNN(dim_latent, dim_input, dim_output).to(device)
optimizer = torch.optim.Adam(modele.parameters(), lr=lr)

# entrainement
for epoch in range(epochs):
    total_loss = 0
    for data in loader:
        data = data.to(device)                # (seq_len, batch)
        #print("tour ")
        inputs  = data[:-1, :]
        targets = data[1:, :]

        optimizer.zero_grad() 

        # entrée du modèle 
        x_emb = modele.embedding_layer(inputs)
        h0 = torch.zeros(data.size(1), modele.Wh.out_features, device=device)

        # forward pass
        H = modele(x_emb, h0)                # (seq_len-1, batch, dim_latent)
        y_pred = modele.decode(H)                # (seq_len-1, batch, vocab_size)

        # Reshape pour la loss
        y_pred = y_pred.view(-1, y_pred.size(-1))
        targets = targets.reshape(-1)

        loss = maskedCrossEntropy(y_pred, targets, PAD_IX)

        # Backprop
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss = {total_loss/len(loader):.4f}")
    writer.add_scalar("Loss/train", total_loss/len(loader), epoch)


# sauvegarde du modele
torch.save(modele.state_dict(), "rnn_trump.pt")
# pour le charger:  rnn.load_state_dict(torch.load("rnn_trump.pt"))

       


"""

#  TODO:  Reprenez la boucle d'apprentissage, en utilisant des embeddings plutôt que du one-hot

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
        
        # x : [batch, length]  --> embedding : [batch, length, dim_input]
        x_embed = model.embedding_layer(x)

        # RNN attend : [length, batch, dim_input]
        x_embed = x_embed.permute(1, 0, 2)

        h0 = torch.zeros(x_embed.size(1), dim_latent).to(device)

        # froward pass dans le RNN
        H = model.forward(x_embed, h0)  # taille [length, batch, dim_latent]
        Y_hat = model.decode(H)         # [length, batch, vocab_size]

        # attention, reshape pour cross entropy: [length * batch, vocab_size]
        loss = criterion(Y_hat.view(-1, vocab_size), y.T.reshape(-1))
        # reshape(-1) -> applati en ligne
        # .view(-1,vocab_size) -> taille [lenght*batch, vocab_size]

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
    writer.add_scalar("Train/loss", total_loss/len(data_trump), epoch)
    writer.add_scalar("Train/accuracy", acc, epoch)
"""
