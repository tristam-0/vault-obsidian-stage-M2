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

## Modification du HungarianMatcher : Intégration de la Focal Loss
il reprend l'utilisation de la focal loss de [[Deformable DETR]]]
> [!info] Liens utiles
> * Concept théorique : [[Focal Loss]]
> * Code source : `models/matcher.py` (Deformable DETR / Anchor DETR)

### Ce qui change :
Dans les évolutions de DETR, le calcul du coût de classification du Matcher passe d'un `softmax` classique à une **Focal Loss** basée sur des `sigmoid` indépendantes .
> [!warning] Changement architectural implicite (Softmax vs Multi-Sigmoïdes)
> L'utilisation de la Focal Loss impose un changement radical dans la manière dont le modèle prédit les classes, bien que ce ne soit pas détaillé explicitement dans le texte du papier :
> 
> * **DETR Original (Softmax) :** Le modèle utilise une unique fonction `softmax` globale. Toutes les classes (y compris la classe "fond") sont en compétition directe au sein d'une distribution de probabilité unique qui doit obligatoirement sommer à $1.0$.
> * **Modèles Modifiés (Multi-Sigmoïdes indépendantes) :** On jette le Softmax. À la place, **chaque classe possède sa propre activation `sigmoid` indépendante**. La détection devient une suite de questions binaires parallèles (ex: *"Est-ce une voiture ? 0.90"*, *"Est-ce un piéton ? 0.01"*). 
> 
> **Les deux conséquences majeures :**
> 1. **Changement de dimension des tenseurs :** On passe d'une couche de sortie de taille **$X + 1$** classes (où le $+1$ était la classe fond dans le DETR original) à une taille de **$X$** classes seulement. 
> 2. **Le fond devient un état implicite :** La classe "fond" (background) n'existe plus en tant que variable ou canal de sortie dans le réseau. Le fond correspond désormais au moment exact où **toutes les sigmoïdes de toutes les classes sont simultanément proches de 0**.
### Pourquoi cette modification ?
* **Stabilisation de l'entraînement :** En calculant le coût via la formule `pos_cost_class - neg_cost_class`, le Matcher évalue l'impact net d'une requête sur la perte globale. Si le modèle est trop incertain, la classification génère une pénalité (coût positif).
* **Élimination du bruit d'assignation :** Tant que le modèle est incertain sur les classes (notamment au début de l'entraînement), le coût positif de la Focal Loss neutralise les micro-fluctuations des probabilités (les bruits à $0.01$ ou $0.02$). L'algorithme hongrois se base alors uniquement sur la **géométrie pure ($L_{\text{box}}$)** pour faire ses choix obligatoires, évitant de sauter d'une requête à l'autre et de déstabiliser la classe "fond" (background).