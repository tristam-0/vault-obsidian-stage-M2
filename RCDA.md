Row-Column Decouple Attention
Le but de cette attention est de sauver de la mémoire. 
## fonctionnement
au lieu de réaliser la tension sur les colonnes et les lignes en même temps. La proposition est de réaliser la tension sur les lignes, puis sur les colonnes. 

idée est de découper la key feature $K_f \in \mathbb{R}^{H \times W \times C}$  (en réalite 2D version flattenée de la grille H×W.) en deux pour avoir la row feature $K_{f,x} \in \mathbb{R}^{W \times C}$  et la column feature $K_{f,x} \in \mathbb{R}^{H \times C}$ en utiliser un opération de 1D global average pooling.

>ex sur la dimension W
Pour chaque ligne (de taille H), on calcule la moyenne de ses éléments à travers toutes les colonnes (de 1 à W). Cela donne un vecteur de taille H×C représentant chaque ligne.  

on difinie $K_{p,x}=G_{1D} (Pos_{k,x})$ et $K_{p,y}=G_{1D} (Pos_{k,y})$ le positionnel encoding sur X ou sur Y. 

on réaliser $K_x=K_{f,x}+K_{p,x}$
##    preuve d'utilité

