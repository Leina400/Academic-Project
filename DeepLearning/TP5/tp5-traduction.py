import logging
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import torch
import unicodedata
import string
from tqdm import tqdm
from pathlib import Path
from typing import List

import time
import re
from torch.utils.tensorboard import SummaryWriter
import random




logging.basicConfig(level=logging.INFO)

FILE = "/home/lilou/Documents/M2/Deep_Learning/TP/data/en-fra.txt"

writer = SummaryWriter("/tmp/runs/tag-"+time.asctime())

def normalize(s):
    return re.sub(' +',' ', "".join(c if c in string.ascii_letters else " "
         for c in unicodedata.normalize('NFD', s.lower().strip())
         if  c in string.ascii_letters+" "+string.punctuation)).strip()


class Vocabulary:
    """Permet de gérer un vocabulaire.

    En test, il est possible qu'un mot ne soit pas dans le
    vocabulaire : dans ce cas le token "__OOV__" est utilisé.
    Attention : il faut tenir compte de cela lors de l'apprentissage !

    Utilisation:

    - en train, utiliser v.get("blah", adding=True) pour que le mot soit ajouté
      automatiquement
    - en test, utiliser v["blah"] pour récupérer l'ID du mot (ou l'ID de OOV)
    """
    PAD = 0
    EOS = 1
    SOS = 2
    OOVID = 3

    def __init__(self, oov: bool):
        self.oov = oov
        self.id2word = ["PAD", "EOS", "SOS"]
        self.word2id = {"PAD": Vocabulary.PAD, "EOS": Vocabulary.EOS, "SOS": Vocabulary.SOS}
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

    def getword(self, idx: int):
        if idx < len(self):
            return self.id2word[idx]
        return None

    def getwords(self, idx: List[int]):
        return [self.getword(i) for i in idx]



class TradDataset():
    def __init__(self,data,vocOrig,vocDest,adding=True,max_len=10):
        self.sentences =[]
        for s in tqdm(data.split("\n")):
            if len(s)<1:continue
            orig,dest=map(normalize,s.split("\t")[:2])
            if len(orig)>max_len: continue
            self.sentences.append((torch.tensor([vocOrig.get(o) for o in orig.split(" ")]+[Vocabulary.EOS]),torch.tensor([vocDest.get(o) for o in dest.split(" ")]+[Vocabulary.EOS])))
    def __len__(self):return len(self.sentences)
    def __getitem__(self,i): return self.sentences[i]



def collate_fn(batch):
    orig,dest = zip(*batch)
    o_len = torch.tensor([len(o) for o in orig])
    d_len = torch.tensor([len(d) for d in dest])
    return pad_sequence(orig),o_len,pad_sequence(dest),d_len

# pad_sequence -> [longueur_sequence , nombre de sequence]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


with open(FILE) as f:
    lines = f.readlines()

lines = [lines[x] for x in torch.randperm(len(lines))]
idxTrain = int(0.8*len(lines))

vocEng = Vocabulary(True)
vocFra = Vocabulary(True)
MAX_LEN=100
BATCH_SIZE=100

datatrain = TradDataset("".join(lines[:idxTrain]),vocEng,vocFra,max_len=MAX_LEN)
datatest = TradDataset("".join(lines[idxTrain:]),vocEng,vocFra,max_len=MAX_LEN)

#print(len(datatrain))
#print(datatrain[0])

train_loader = DataLoader(datatrain, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(datatest, collate_fn=collate_fn, batch_size=BATCH_SIZE, shuffle=True)

#or batch in train_loader :
  #  print(batch)
   # break

#  TODO:  Implémenter l'encodeur, le décodeur et la boucle d'apprentissage

class Encodeur(nn.Module):
    def __init__(self, src_vocab_size, hidden_dim, embed_dim):
        super().__init__()
        self.emb = nn.Embedding(src_vocab_size, embed_dim, padding_idx = Vocabulary.PAD)
        self.gru = nn.GRU(embed_dim, hidden_dim)

    def forward(self, src_pad, src_len):
        seq_emb = self.emb(src_pad)
        # de taille [longueur src, B, H]
        out, ht = self.gru(seq_emb)
        # out : les etats a chaque pas de temps [longueur src, B, H]
        # ht le dernier eetat [1,B,H] a transmettre au decodeur (forme standar de sortie quand num_layer = 1)
        return ht 


class Decodeur(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.emb = nn.Embedding(target_vocab_size, embed_dim, padding_idx = Vocabulary.PAD)
        self.gru = nn.GRU(embed_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, target_vocab_size)

    # on transmet le dernier etat caché au décodeur 
    def step(self, y, h):
        # y taille [B] -> batch apres emb -> [B, E] et on veut que GRU lit un seul mot pour generer un seul mot 
        # donc on veut [1,B,E] avec 1 = T la taille de la séquence 
        emb = self.emb(y).unsqueeze(0)
        out, h = self.gru(emb, h)
        # on veut [1,B,H] -> [B,H] et la couche lineare attend en entree [B,H] oour oroduire [B, vocab_size]
        logits = self.proj(out.squeeze(0))
        return logits, h

    
    def generate_one(self, hidden, lenseq=None, max_len=100,
                     sos_id=Vocabulary.SOS, eos_id=Vocabulary.EOS):
        """
        hidden: [1,1,H]  -> un seul exemple
        """
        device = hidden.device
        T = int(lenseq) if (lenseq is not None) else max_len

        y = torch.tensor([sos_id], device=device, dtype=torch.long)  # [1]
        outputs = []

        for _ in range(T):
            logits, hidden = self.step(y, hidden)  # logits: [1,|V|] # [0.1, 0.06, ..] de al taille du vocabulaire
            y = logits.argmax(dim=-1)              # [1]
            outputs.append(y.item())               
            if y.item() == eos_id:                 # son verifie si le mot qu'on vient de faire si cest EOS ou pas
                break

        return torch.tensor(outputs, device=device, dtype=torch.long)  # [T_gen]
    
    def generate(self, hidden, lenseq=None, max_len=100,
                 sos_id=Vocabulary.SOS, eos_id=Vocabulary.EOS):
        """
        hidden: [1,B,H]
        retour: [T_max, B] après padding avec EOS
        """
        B = hidden.size(1) # on redcupere la dim2
        seqs = []
        for b in range(B):
            # isole l'état caché du b-ième exemple doit etre de la forme [1,1,H]
            h_b = hidden[:, b, :].unsqueeze(1)
            seq_b = self.generate_one(h_b, lenseq=lenseq, max_len=max_len,
                                      sos_id=sos_id, eos_id=eos_id)  # [T_b]
            seqs.append(seq_b)

        # pad en hauteur avec EOS pour aligner: résultat [T_max, B]
        return pad_sequence(seqs, batch_first=False,
                            padding_value=eos_id)


# boucle d'apprentissage 
# pour parcourir train_dataset , collate_fn renvoie pad_sequence(orig),o_len,pad_sequence(dest),d_len 
# (pad_src, len_src, pad_targ, len_targ)

HIDDEN = 64
EMBED = 32

encodeur = Encodeur(len(vocEng), hidden_dim = HIDDEN, embed_dim = EMBED)
decodeur = Decodeur(len(vocFra), embed_dim = EMBED, hidden_dim = HIDDEN)
criterion = nn.CrossEntropyLoss(ignore_index = Vocabulary.PAD)
opti = torch.optim.Adam(list(encodeur.parameters())+ list(decodeur.parameters()))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EPOCHS = 10

def train():
    print("début de l'entrainement ")
    for epoch in range(1, EPOCHS+1):
        encodeur.train()
        decodeur.train()
        total_loss = 0

        for src_pad, src_len, targ_pad, targ_len in train_loader :
            src_pad, targ_pad = src_pad.to(device), targ_pad.to(device)

            opti.zero_grad()

            # tirage aleatoire du mode :
            mode_constraint = random.random() < 0.5 

            # encodage 
            h = encodeur(src_pad, src_len)

            # initalisation : y = [SOS, SOS, ..., SOS] de taille le nombre de phrase , pour chaque phrase, ca commence a SOS
            B = targ_pad.size(1)
            y = torch.full((B,), Vocabulary.SOS, dtype=torch.long, device=device)
            loss = 0

            if mode_constraint :
                # teacher forcing -> phrase cible est passé au decodeur et a chasue pas de temps cest un mot de la phrase cible
                # qui est considér
                # pour chaque phrase cible 
                for t in range(targ_pad.size(0)):
                    logits, h = decodeur.step(y, h)
                    y = targ_pad[t]
                    # y a la forme [B], cst le token cible au pas de temps t 
                    loss += criterion(logits, y)

            else :
                # mode non constraint -> mot correspondant a la probabilité max du decodage du pas precedant est donne au decodeur
                # on genere a aprtir des predictions
                T = int(targ_len.max().item())  # on itere sur toutes les phrases cibles (on prend la longueur max et de toute facon elles sont paddées)
                for t in range(T):
                    logits, h = decodeur.step(y,h)
                    y = logits.argmax(-1) # mot prédit distrib sur les V mots possible du vocabulaire [B,V] ok pour cross entreopy pour les taille
                    loss += criterion(logits, targ_pad[t])

            loss = loss / targ_pad.size(0)  # moyenne sur la sequence
            loss.backward()
            opti.step()

            total_loss += loss.item()

        encodeur.eval()
        decodeur.eval()

        with torch.no_grad():
            test_loss = 0

            for src_pad, src_len, targ_pad, targ_len in test_loader:
                src_pad, targ_pad = src_pad.to(device), targ_pad.to(device)

                h = encodeur(src_pad, src_len)
                B = targ_pad.size(1)
                y = torch.full((B,), Vocabulary.SOS, dtype=torch.long, device=device)
                loss = 0

                for t in range(targ_pad.size(0)):
                    logits, h = decodeur.step(y, h)
                    y = targ_pad[t]
                    loss += criterion(logits, targ_pad[t])
                test_loss += (loss / targ_pad.size(0)).item()

        print(f"Epoch {epoch+1}/{EPOCHS} | Train loss: {total_loss/len(train_loader):.4f} | Test loss: {test_loss/len(test_loader):.4f}")
        writer.add_scalar("Train/Loss", total_loss/len(train_loader), epoch)
        writer.add_scalar("Test/Loss", test_loss/len(test_loader), epoch)

        torch.save({
        'encodeur': encodeur.state_dict(),
        'decodeur': decodeur.state_dict(),
        }, 'modele_trad.pth')

#train()

"""
point sur les tailles :
src_pad : sequences d'entrees paddées [T_src_max, B] T_src_max : longueur max des phrases, B = batch (nombre de phras)
src_len : longueur reelle de chaque sequence source [B] 
targ_pad : sequence cibles padees [T_targ_max, B], T_targ_max : lonugeur max des phrase cibles, B : nombre de phrases
targ_len : [B] longueur de cjaque phrase cible


logits = tensor([
  [ 2.0,  0.1, -1.0,  0.5, -0.2],   # phrase 1
  [-1.0,  2.3,  0.5, -0.2,  1.1],   # phrase 2
  [ 0.3,  0.2,  0.1,  1.0, -0.5],   # phrase 3
])  # shape [B=3, V=5]

"""

# visualiser les traudctions proposés : 

checkpoint = torch.load("modele_trad.pth", map_location=device)

encodeur.load_state_dict(checkpoint["encodeur"])
decodeur.load_state_dict(checkpoint["decodeur"])
encodeur.to(device).eval()
decodeur.to(device).eval()

def traduire(phrase):
    # on prepare la phrase d'netre 
    mots = normalize(phrase).split(" ")
    ids = [vocEng.get(w, adding=False) for w in mots] + [Vocabulary.EOS]
    src = torch.tensor(ids).unsqueeze(1).to(device)   # [T,1]
    src_len = torch.tensor([len(ids)]).to(device)

    # passe les token dans l=encodeur et decodeur
    with torch.no_grad():
        h = encodeur(src, src_len)
        out = decodeur.generate(h, lenseq=20)         # [T_gen,1]

    # convertit en mots pour que ce soit interpretable
    trad = [vocFra.getword(i) for i in out.squeeze().tolist()]
    trad = [w for w in trad if w not in ("PAD","SOS","EOS",None)]
    return " ".join(trad)

# exemple  (tres mauvais aleatoire)
print(traduire("i am happy"))
print(traduire("i am angry"))
print(traduire("nice to meet you"))
