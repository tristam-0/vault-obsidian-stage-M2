Row-Column Decouple Attention
Le but de cette attention est de sauver de la mémoire. 
Elle est utilisée au niveau de la **cross attention** du décodeur et diminue également la computational complexity. 
## fonctionnement
au lieu de réaliser la tension sur les colonnes et les lignes en même temps. La proposition est de réaliser attentions sur les lignes, puis sur les colonnes. 
**Attention** : on fais  hypotheses que W>H pour l'ordre de équitation
**ATTENTION NOTATION** : C est la dimension totale du vecteur. Les auteurs semblent avoir choisi de l'utiliser partout pour représenter ce qui arrive dans l'ensemble des têtes. Si on veut avoir un point de vue mono-tête, il faut remplacer C par $d_k$ . 

initialement on a ${H \times W \times C}$ features qui sont extraites d'un CNN.
puis que on a réduit a un matrice $X$ de taille  ${H W \times C}$.
On fait passer la matrice X à travers les blocks de l'encoder puis a travers 2 matrice $W_k$, $W_v$ pour obtenir 2 matrices $K_f,V_f$
![[RCDA-1778507062709.webp]]
On a Key feature représentation de $K_f \in \mathbb{R}^{H \times W \times C}$  pour découpler la ligne et la colonne on va réaliser une opération 1D global average pooling et ainsi obtenir la row feature $K_{f,x} \in \mathbb{R}^{W \times C}$  et la column feature $K_{f,y} \in \mathbb{R}^{H \times C}$

>ex sur la dimension W
Pour chaque ligne (de taille H), on calcule la moyenne de ses éléments à travers toutes les colonnes (de 1 à W). Cela donne un vecteur de taille H×C représentant chaque ligne.  

![[RCDA-1778507876440.webp|462x312]]
### positionnelle encoding
on définie $K_{p,x}=g_{1D} (Pos_{k,x})$ et $K_{p,y}=g_{1D} (Pos_{k,y})$ le positionnel encoding 1D sur X ou sur Y. 
on réaliser $K_x=K_{f,x}+K_{p,x}$ et $K_y=K_{f,y}+K_{p,y}$

on n'a $Q_f$ qui vien de la seff attantion de décodeur de taill $K_f \in \mathbb{R}^{N_q \times C}$ avec $N_q$ le nombre de Anchor point.
![[RCDA-1778509200492.webp|396x379]]
on réaliser le positionnelle encoding $g$ sur une dimension $Q_{p,x}=g_{1D}(pos_{q,x})$ puis sur l'autre $Q_{p,y}=g_{1D}(pos_{q,y})$ et on obtient en deux encodages de position 1D distincts.  
$Q_x=Q_{x}+Q_{p,x}$ et $Q_y=Q_{f}+Q_{p,y}$

Ensuite, on calcule les **scores d’alignement** pour les row et les column. On obtient ainsi deux matrices $A_x \in \mathbb{R}^{N_q \times W}$ et $A_y \in \mathbb{R}^{N_q \times H}$.
qui représentent la tension sur les lignes ou sur les colonnes. 
![[RCDA-1778510411787.webp]]
![[RCDA-1778569938880.webp]]
Il ne nous reste donc plus qu'à les combiner. 
L'agrégation se fait en deux passes successives (cascade) pour éviter de reconstruire la matrice d'attention complète $H \times W$

on a $V \in \mathbb{R}^{H \times W \times C}$
$Z=Weighted_{sumW}(A_x,V),Z\in \mathbb{R}^{N_q \times H \times C}$ 
$Z_{n, h, c} = \sum_{w=1}^{W} A_{x_{n, w}} \cdot V_{h, w, c}$ 
![[RCDA-1778582188321.webp|697|700x202]]
$Out=Weighted_{sumH}(A_y,Z),Out\in \mathbb{R}^{N_q \times C}$ 
$Output_{n, c} = \sum_{h=1}^{H} A_{y, n, h} \cdot Z_{n, h, c}$
![[RCDA-1778582345476.webp]]

### biais de la méthode
>Le découpage de l'attention en deux étapes indépendantes ($A_x$ puis $A_y$) impose une contrainte structurelle forte : pour qu'un pixel de l'image situé à la coordonnée $(x, y)$ reçoive une attention élevée, il faut **simultanément** que la Query considère que la colonne $x$ est importante (score élevé dans $A_x$) ET que la ligne $y$ est importante (score élevé dans $A_y$). 

Ce biais force le modèle à focaliser son attention sur l'intersection d'une ligne et d'une colonne, créant une **croix de sélection**
##    Preuve d'utilité
### Computational complexity
En s'affranchissant de l'utilisation d'une matrice d'attention globale de taille H×W, le modèle réduit la complexité algorithmique globale. 
On passe ainsi d'une dépendance quadratique par rapport à la résolution de l'image, soit $\mathcal{O}(N_q \cdot (H + W) \cdot C)$, à une dépendance linéaire, soit $\mathcal{O}(N_q \cdot H \cdot W \cdot C)$. 
Bien que l'article _Anchor DETR_ se contente de mentionner cette réduction, le résultat peut être vérifié en comparant le coût de calcul de deux opérations _Softmax_ bidimensionnelles découplées (sur la hauteur et la largeur) face au coût d'un unique _Softmax_ global appliqué à l'ensemble des pixels dans l'implémentation classique.
### Memory save
le coup principale de memory est l'attention weight map normalement $A \in \mathbb{R}^{N_q \times H \times W \times M}$  
avec M le nombre de head
mais dans RDCA on la diviser en 2 , $A_x \in \mathbb{R}^{N_q \times W \times C \times M}$  et $A_y \in \mathbb{R}^{N_q \times H \times C \times M}$
et l'étape qui demande le plus de mémoire est donc  $z\in \mathbb{R}^{N_q \times H \times C}$
donc le ratio de memory save devient 
$$r = \frac{N_q \times H \times W \times M}{N_q \times H \times C} = \frac{W \times M}{C}$$
### Exemple
Considérons une configuration avec $M = 8$ têtes d'attention et une dimension de canal $C = 256$. Pour une image de taille originale $640 \times 480$ pixels, le passage par le _backbone_ de DETR réduit la résolution spatiale par un facteur 32, ce qui donne une largeur de carte de caractéristiques $W = 20$.
Dans ce cas, le ratio d'économie de mémoire est :
$r = \frac{20 \times 8}{256} = 0,625$
À ce stade, la méthode ne permet pas d'économiser de la mémoire (elle en consomme même davantage). Cependant, le gain devient significatif lorsque la résolution augmente :
- **Si $W = 32$** : $r = \frac{32 \times 8}{256} = 1$
- **Si $W = 40$** : $r = \frac{40 \times 8}{256} = 1,25$
- **Si $W = 80$**  : $r = \frac{80 \times 8}{256} = 2,5$ 
