Row-Column Decouple Attention
Le but de cette attention est de sauver de la mémoire. 
Elle est utilisée au niveau de la **cross attention** du décodeur et diminue également la computational complexity. 
## fonctionnement
au lieu de réaliser la tension sur les colonnes et les lignes en même temps. La proposition est de réaliser attentions sur les lignes, puis sur les colonnes. 

initialement on a ${H \times W \times C}$ features qui sont extraites d'un CNN.
puis que on a réduit a un matrice $X$ de taille  ${H W \times C}$.
On fait passer la matrice X à travers les blocks de l'encoder puis a travers 2 matrice w_k,w_v,. pour obtenir 2 matrices $K_f,V_f$
![[RCDA-1778507062709.webp]]
On a Key feature représentation de $K_f \in \mathbb{R}^{H \times W \times C}$  pour découpler la ligne et la colonne on va réaliser une opération 1D global average pooling et ainsi obtenir la row feature $K_{f,x} \in \mathbb{R}^{W \times C}$  et la column feature $K_{f,x} \in \mathbb{R}^{H \times C}$

>ex sur la dimension W
Pour chaque ligne (de taille H), on calcule la moyenne de ses éléments à travers toutes les colonnes (de 1 à W). Cela donne un vecteur de taille H×C représentant chaque ligne.  

![[RCDA-1778507876440.webp|462x312]]
positionnelle encoding

on définie $K_{p,x}=g_{1D} (Pos_{k,x})$ et $K_{p,y}=g_{1D} (Pos_{k,y})$ le positionnel encoding 1D sur X ou sur Y. 
on réaliser $K_x=K_{f,x}+K_{p,x}$ et $K_y=K_{f,y}+K_{p,y}$

on n'a $Q_f$ qui vien de la seff attantion de décodeur de taill $K_f \in \mathbb{R}^{N_q \times C}$ avec $N_q$ le nombre de Anchor point
![[RCDA-1778509200492.webp|396x379]]
on réaliser le positionnelle encoding sur une dimension puis sur l'autre et on obtient ainsi deux query qui dépendent de la dimension.  
$Q_x=Q_{x}+Q_{p,x}$ et $Q_y=Q_{f}+Q_{p,y}$

Ensuite, on calcule les scores d’alignement pour les row et les column. On obtient ainsi deux matrices $A_x \in \mathbb{R}^{N_q \times W}$ et $A_y \in \mathbb{R}^{N_q \times H}$.
qui représentent la tension sur les lignes ou sur les colonnes. 
![[RCDA-1778510411787.webp]]
![[RCDA-1778569938880.webp]]
on a $V \in \mathbb{R}^{H \times W \times C}$
$Z=Weighted_{sumW}(A_x,V),Z\in \mathbb{R}^{N_q \times H \times C}$ 

##    preuve d'utilité

