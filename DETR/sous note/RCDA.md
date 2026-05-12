Row-Column Decouple Attention
Le but de cette attention est de sauver de la mémoire. 
## fonctionnement
au lieu de réaliser la tension sur les colonnes et les lignes en même temps. La proposition est de réaliser attentions sur les lignes, puis sur les colonnes. 

initialement on a ${H \times W \times C}$ features qui sont extraites d'un CNN.
puis que on a rédui a un matrice $X$ de taille  ${H W \times C}$.
On fait passer la matrice X à travers 3 matrice w_k,w_v,w_q. pour obtenir trois matrices $K_f,V_f,Q_f$ 
![[RCDA-1778507062709.webp]]
donc on a Key feature représentation de $K_f \in \mathbb{R}^{H \times W \times C}$  pour découpler la ligne et la colonne on va réaliser une opération 1D global average pooling et ainsi obtenir la row feature $K_{f,x} \in \mathbb{R}^{W \times C}$  et la column feature $K_{f,x} \in \mathbb{R}^{H \times C}$

>ex sur la dimension W
Pour chaque ligne (de taille H), on calcule la moyenne de ses éléments à travers toutes les colonnes (de 1 à W). Cela donne un vecteur de taille H×C représentant chaque ligne.  

![[RCDA-1778507876440.webp]]
positionnelle encoding

on définie $K_{p,x}=g_{1D} (Pos_{k,x})$ et $K_{p,y}=g_{1D} (Pos_{k,y})$ le positionnel encoding 1D sur X ou sur Y. 
on réaliser $K_x=K_{f,x}+K_{p,x}$ et $K_y=K_{f,y}+K_{p,y}$
![[RCDA-1778509200492.webp]]
on réaliser $Q_x=Q_{x}+Q_{p,x}$ et $Q_y=Q_{f}+Q_{p,y}$

Ensuite, on calcule les scores d’alignement pour les row et les column. On obtient ainsi deux matrices $A_x$ et $A_y$.
![[RCDA-1778510411787.webp]]

##    preuve d'utilité

