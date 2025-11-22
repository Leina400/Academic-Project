from textloader import  string2code, id2lettre
import math
import torch

#  TODO:  Ce fichier contient les différentes fonction de génération

@torch.no_grad()
def generate(rnn, emb, decoder, eos, start="", deterministic = True, maxlen=200, device="cpu"):
    """  
        Fonction de génération (l'embedding et le decodeur être des fonctions du rnn). Initialise le réseau avec start (ou à 0 si start est vide) et génère une séquence de longueur maximale 200 ou qui s'arrête quand eos est généré.
        * rnn : le réseau
        * emb : la couche d'embedding
        * decoder : le décodeur
        * eos : ID du token end of sequence
        * start : début de la phrase
        * maxlen : longueur maximale
    """

    # passage en mode evaluation pour descatvier dropout etc
    rnn.eval()
    rnn.to(device)

    # init du modele
    batch_size = 1
    h = torch.zeros(batch_size, rnn.Wh.out_features, device=device)

    # preprocessing pour start, le texte de départ
    if start != "":
        # on le transofmre en des entiers
        start_input = string2code(start).to(device)
        for ch in start_input:
            # pour chaque caractère, on les  envoie dans l'embedding layer
            x = rnn.embedding_layer(ch.unsqueeze(0))  # [1, dim_output]
            # on fait un pas 
            h = rnn.one_step(x, h)
        last_char = start_input[-1].unsqueeze(0)
    else:
        last_char = torch.tensor([0], dtype=torch.long, device=device)  # PAD_IX

    generated_indices = []

    # boucle de generation 
    for _ in range(maxlen):
        # embedding du dernier caractère
        x = rnn.embedding_layer(last_char)
        # calcul de l'état caché suivant
        h = rnn.one_step(x, h)
        # decodage de la sortie
        probs = rnn.decode(h)  # [1, vocab_size]

        if deterministic:
            next_char = torch.argmax(probs, dim=-1)
        else:
            next_char = torch.multinomial(probs.squeeze(0), 1)

        next_char_idx = next_char.item()
        generated_indices.append(next_char_idx)

        if next_char_idx == eos:
            break

        last_char = next_char

    # reconstruction de la séquence
    generated_text = ''.join(id2lettre[i] for i in generated_indices)
    return start + generated_text



def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200):
    """
        Génere une séquence en beam-search : à chaque itération, on explore pour chaque candidat les k symboles les plus probables; puis seuls les k meilleurs candidats de l'ensemble des séquences générées sont conservés (au sens de la vraisemblance) pour l'itération suivante.
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
    
    
    
    
 # ======= brouillon pour beam search a re tester car pb
 """
 
def generate_beam(rnn, emb, decoder, eos, k, start="", maxlen=200):
    rnn.eval()

    # init etat caché
    h = torch.zeros(1, rnn.Wh.out_features)

    start_ids = string2code(start)
    for ch in start_ids:
       x = emb(ch.unsqueeze(0))
       h = rnn.one_step(x, h)
    last_char = start_ids[-1].unsqueeze(0)

    
    #  liste de tuples
    beam = [([], 0.0, h, last_char)]

    for i in range(maxlen):
        new_beam = []

        for seq, logp, h_prev, last_c in beam:
            # un pas de RNN
            x = emb(last_c)
            h_new = rnn.one_step(x, h_prev)

            # proba scores
            logits = decoder(h_new)
            log_probs = torch.log_softmax(logits, dim=-1).squeeze(0)

           # k meilleures
            top_logp, top_idx = torch.topk(log_probs, k)

            # Pour chaque symbole on cree une nouvelle seq
            for lp, idx in zip(top_logp, top_idx):
                new_seq = seq + [idx.item()]
                new_logp = logp + lp.item()
           

        # On garde les k meilleures séquences du beam
        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:k]

    # meilleure séquence finale
    best_seq = beam[0][0]
    txt = ''.join(id2lettre[i] for i in best_seq if i != eos)  # mettre en interpetabe
    return start + txt


#On garde seulement les caractères cumulant a de masse de probabilité.

# a tester
def p_nucleus(decoder, alpha):
    def compute(h):
        logits = decoder(h)
        probs = torch.softmax(logits, dim=-1).squeeze(0)

        # tri décroissant
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)

        # somme cumulée
        cumulative = torch.cumsum(sorted_probs, dim=0)

        # plus petit ensemble couvrant a
        cutoff = (cumulative >= alpha).nonzero(as_tuple=True)[0][0]

        # Indices autorisés
        keep = sorted_idx[:cutoff+1]

        # construire nouvelle distribution
        new_probs = torch.zeros_like(probs)
        new_probs[keep] = probs[keep]
        new_probs /= new_probs.sum()

        return new_probs.unsqueeze(0)  # shape (1, vocab)

    return compte 
    
    """
 
