import itertools
import logging
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
from typing import List
import time
from conllu import parse_incr
logging.basicConfig(level=logging.INFO)
import datetime

writer = SummaryWriter("runs/TP5Tagging"+datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
DATA_PATH = "/home/lilou/Documents/M2/Deep_Learning/TP/data/"


# Format de sortie décrit dans
# https://pypi.org/project/conllu/

class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement s'il n'est pas connu
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    OOVID = 1
    PAD = 0

    def __init__(self, oov: bool):
        """ oov : autorise ou non les mots OOV """
        self.oov =  oov
        self.id2word = [ "PAD"]
        self.word2id = { "PAD" : Vocabulary.PAD}
        if oov:
            self.word2id["__OOV__"] = Vocabulary.OOVID
            self.id2word.append("__OOV__")

    def __getitem__(self, word: str):
        if self.oov:
            return self.word2id.get(word, Vocabulary.OOVID)
        return self.word2id[word]

    def get(self, word: str, adding=True):
        try:
            return self.word2id[word]
        except KeyError:
            if adding:
                wordid = len(self.id2word)
                self.word2id[word] = wordid
                self.id2word.append(word)
                return wordid
            if self.oov:
                return Vocabulary.OOVID
            raise

    def __len__(self):
        return len(self.id2word)

    def getword(self,idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self,idx: List[int]):
        return [self.getword(i) for i in idx]



class TaggingDataset():
    def __init__(self, data, words: Vocabulary, tags: Vocabulary, adding=True):
        self.sentences = []

        for s in data:
                self.sentences.append(([words.get(token["form"], adding) for token in s], [tags.get(token["upostag"], adding) for token in s]))
    def __len__(self):
        return len(self.sentences)
    def __getitem__(self, ix):
        return self.sentences[ix]


def collate_fn(batch):
    """Collate using pad_sequence"""
    return tuple(pad_sequence([torch.LongTensor(b[j]) for b in batch]) for j in range(2))

logging.info("Loading datasets...")
words = Vocabulary(True)
tags = Vocabulary(False)

data_file = open(DATA_PATH+"fr_gsd-ud-train.conllu",encoding="utf-8")
train_data = TaggingDataset(parse_incr(data_file), words, tags, True)

data_file = open(DATA_PATH+"fr_gsd-ud-dev.conllu",encoding='utf-8')
dev_data = TaggingDataset(parse_incr(data_file), words, tags, True)

data_file = open(DATA_PATH+"fr_gsd-ud-test.conllu",encoding="utf-8")
test_data = TaggingDataset(parse_incr(data_file), words, tags, False)


logging.info("Vocabulary size: %d", len(words))

BATCH_SIZE=100

train_loader = DataLoader(train_data, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
dev_loader = DataLoader(dev_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_data, collate_fn=collate_fn, batch_size=BATCH_SIZE)



#exemple = train_data[0]
#print("Tokens (indices) :", exemple[0])
#print("Tags (indices)   :", exemple[1])

# Pour voir les mots et tags correspondants :
#print("Mots :", words.getwords(exemple[0]))
#print("Tags :", tags.getwords(exemple[1]))



#  TODO:  Implémenter le modèle et la boucle d'apprentissage (en utilisant les LSTMs de pytorch)


# Entee : une sequence de tokens de longueur T
# Sortie : une sequence de tags de même longueur T
# Chaque token doit correspond a une classe grammaticale.

class modele(nn.Module):
    def __init__(self, vocab_size, tagset_size, emb_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=Vocabulary.PAD)
        self.lstm = nn.LSTM(emb_dim, hidden_dim) # ht = f(xt,ht-1)
        self.fc = nn.Linear(hidden_dim, tagset_size) # fonction de decodage y = d(ht)

    def forward(self, x):
        # x : [seq_len, batch]
        x_emb = self.embedding(x)  # [seq_len, batch, emb_dim] a trasformé chaque token en vecteur continue
        ht, _ = self.lstm(x_emb)                 # [seq_len, batch, hidden_dim]
        yt = self.fc(ht)                      # [seq_len, batch, n_tags]
        return yt


# Boucle d'apprentissage pour le tagging

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = modele(len(words), len(tags), emb_dim=128, hidden_dim=256).to(device)
eps = 0.001
opti = torch.optim.Adam(model.parameters(), lr = eps)
criterion = torch.nn.CrossEntropyLoss(ignore_index = Vocabulary.PAD)
EPOCHS = 10
p_oov = 0.1


def boucle_tagging() :
    best_dev = float('inf')  
    for epochs in range(1, EPOCHS+1):

        model.train()
        total_loss_num = 0.0      # somme perte  par tokens valides
        total_tokens   = 0        # tokens valides
        total_correct  = 0        # prédictions correctes sur tokens valides

        for (token,tag_) in train_loader:
            token = token.to(device)
            tag_ = tag_.to(device)

            if p_oov > 0:
                with torch.no_grad():
                    # on cree un tenseur de meme taille que token composé de chiffre aleatoire entre 0 et 1
                    rnd = torch.rand_like(token, dtype=torch.float)
                    # tenseur m ou ses cases  sont TRue si case(rnd) < p_oov et la ou on est pas sur du padding
                    m = (rnd < p_oov) & (token != Vocabulary.PAD)
                    # on remplace les cases correspondant de token par "OOVID"
                    # remplace certains éléments du tenseur selon un masque par l'element vocabulary.OOVID
                    token = token.masked_fill(m, Vocabulary.OOVID)

            opti.zero_grad()

            logits = model(token)
            T,B,C = logits.shape

            loss = criterion(logits.view(T*B,C), tag_.view(T*B))

            loss.backward()
            opti.step()

            # calcul métrique
            with torch.no_grad():
                mask = (tag_ != Vocabulary.PAD)            # [T,B]
                pred = logits.argmax(-1)                    # [T,B]
                correct = ((pred == tag_) & (mask == True)).sum().item()
                valid   = mask.sum().item()

                total_correct  += correct
                total_tokens   += valid
                total_loss_num += loss.item() * valid       # moyenne pondéerés 

        train_loss = total_loss_num / max(total_tokens, 1)  # evite la division par 0
        train_acc  = total_correct  / max(total_tokens, 1)

        model.eval()
        dev_loss_num = 0.0
        dev_tokens   = 0
        dev_correct  = 0

        # batch de validations
        with torch.no_grad():
            for token, tag_ in dev_loader:
                token = token.to(device)
                tag_  = tag_.to(device)

                logits = model(token)                      # [T,B,C]
                T, B, C = logits.shape
                loss = criterion(logits.view(T*B, C), tag_.view(T*B))

                mask = (tag_ != Vocabulary.PAD)
                pred = logits.argmax(-1)
                dev_correct += ((pred == tag_) & (mask == True)).sum().item()
                valid = mask.sum().item()
                dev_tokens += valid
                dev_loss_num += loss.item() * valid

        dev_loss = dev_loss_num / max(dev_tokens, 1)
        dev_acc  = dev_correct  / max(dev_tokens, 1)

        if dev_loss < best_dev:
            best_dev = dev_loss
            torch.save(model.state_dict(), "best_tagger.pt")


        writer.add_scalar("Loss/train", train_loss, epochs)
        writer.add_scalar("acc/train", train_acc, epochs)
        writer.add_scalar("LOss/val", dev_loss, epochs)
        writer.add_scalar("Acc/val", dev_acc, epochs)
        print(f"epochs {epochs:02d}/{EPOCHS} | train loss {train_loss:.4f} acc_train {train_acc:.4f} | dev loss {dev_loss:.4f} acc_ev {dev_acc:.4f}")

            
#boucle_tagging()


model.load_state_dict(torch.load("best_tagger.pt", map_location=device))
model.eval()
i = 0
tokens_ids, _ = test_data[i]

# transforme en tenseur dans la bonne taille [T,1]
x = torch.tensor(tokens_ids, dtype=torch.long, device=device).unsqueeze(1)

logits = model(x)
pred = logits.argmax(-1).squeeze(1).cpu().tolist()

mots = words.getwords(tokens_ids)
tags_predits = tags.getwords(pred)

# affichage couples 
for mot, tag in zip(mots, tags_predits):
    print(f"{mot:15s} -> {tag}")


# ok fonctionne 