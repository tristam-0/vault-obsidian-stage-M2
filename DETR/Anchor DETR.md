https://arxiv.org/pdf/2109.07107
Dans l’architecture de base de **DETR**, chaque object query peut finir par couvrir une région assez large de l’image.  
Le problème est que ces object queries ne sont pas explicitement contraintes à se concentrer sur une zone précise, ce qui rend l’optimisation plus difficile.
Dans l'article, il propose l'utilisation des **anchor points** afin de guider chaque object query vers une position spatiale plus localisée.
![[Pasted image 20260507141537.png]]

## Anchor Points

Un **anchor point** correspond à une position (x,y) sur la feature map, avec des coordonnées normalisées entre 0 et 1.
Ces points peuvent être :
- initialisés sur une grille,
- initialisés aléatoirement, puis appris pendant l’entraînement.
![[Pasted image 20260507141729.png]]
L’idée est que chaque **object query** soit associé à un point d’ancrage, ce qui rend son rôle plus interprétable.
![[Anchor DETR-1778595133002.webp]]
## Anchor Points
### Attention dans DETR
$$ \text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
$$ Q = Q_f + Q_p, \quad K = K_f + K_p, \quad V = V_f $$
**Souscriptions** : `_f` = feature, `_p` = position embedding
#### Self-Attention (Decoder)
- $Q_f, K_f, V_f$ = sortie decoder précédent
- $Q_p = K_p$ = learned embedding (DETR original) :
$$ Q_p = \text{Embedding}(N_q, C)$$
#### Cross-Attention (Decoder)
- $Q_f$ = sortie self-attention
- $K_f, V_f$ = features encodeur $\mathbb{R}^{HW \times C}$
- $K_p$ = sinus-cosinus sur positions clés :
$$
K_p = g_{\sin}(\text{Pos}_k), \quad \text{Pos}_k \in \mathbb{R}^{HW \times 2}
$$
### Anchor Points → Object Query
**Problème learned $Q_p$** : pas interprétable, pas localisé
**Solution DAB** : remplacer  learned embedding par un positional encoding les **anchor points** que on note $\text{Pos}_q \in \mathbb{R}^{N_A \times 2}$ qui corespondent  au $N_a$ anchor points avec leur (x,y) positon:
$$
Q_p = \text{Encode}(\text{Pos}_q)
$$
**Encoding partagé** (même que clés) :
$$
Q_p = g(\text{Pos}_q), \quad K_p = g(\text{Pos}_k)
$$
**g  [[Positional Encoder]] fonction** :  Dans l'article, il est décide d'utiliser un  MLP 2 couches (pas $g_{\sin}$ heuristique)
### Multiple Predictions par Anchor
**Problème** : Un seul anchor point peut couvrir une région contenant plusieurs objets. Une seule object query par anchor ne suffit pas à les détecter tous.
**Naïve solution** (incorrecte) : Dupliquer $N_p$ object queries identiques sur le même anchor point.  
**Problème** : Toutes les queries identiques reçoivent les mêmes features et attention weights → mêmes prédictions.
**Solution au problème de la solution naïve.** : Associer $N_p$ (Un petit nombre, par ex $N_p=3$. ) **pattern embeddings** distincts à chaque anchor point.
$$ Q^i_f = \text{Embedding}(N_p, C) \quad (\text{patterns partagés}) $$
**Construction de la query initiale** (premier decoder layer) :
$$ Q^{init} = Q_p +Q^i_f $$
**Decoder layers suivants** : self-attention classique avec $Q_f$​ = sortie de la couche précédente.
**Résultat** : Chaque anchor point génère $N_p=3$ slots localisés et différent autour de sa position spatiale

![[Pasted image 20260507154136.png]]
## Row-Column Decoupled Attention
![[RCDA]]