Cet article propose des modification de architecture [[DETR]]
## Amélioration par rapport à DETR. 
amélioration des performances sur les petits objets et meilleures performances obtenues en 10 fois moins d'entraînement 

## origine de l'idee
![[Deformable convolution]]
## déformable Attention module
propose une modification du calcule de l'attantion
de
$\LARGE \mathrm{MultiHeadAttn}(z_q, x) = \sum_{m=1}^{M} W_m \left[ \sum_{k \in \Omega_k} A_{mqk} \cdot W'_m x_k \right]$

$\LARGE \mathrm{DeformAttn}(z_q, p_q, x) = \sum_{m=1}^{M} W_m \left[ \sum_{k=1}^{K} A_{mqk} \cdot W'_m x\bigl(p_q + \Delta p_{mqk}\bigr) \right]$
- $q$ : indice de la requête.
- $m$ : indice de la tête d’attention.
- k indexes the sampled keys, and K is the total sampled key
number (K << HW ) # meme choise que N (object queries dans DETR ? )
- $∆p_{mqk}$ and $A_{mqk}$ denote the sampling offset and attention weight of the
kth sampling point in the mth attention head, respectively
- the scalar attention weight $A_{mqk}$ lies in the range [0, 1], normalized by $∑^K_{k=1} A_{mqk} = 1$ # scalar attention ? je croier que One head Attention = $sofmax({Q*K^t}/\sqrt{d_k})*V =Z$ puis on concaténe les Z et on multiplie par une autre matrice pour combinais les diférante téte besoin explication ou sa vien.
