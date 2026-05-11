vision attention
[[Deformable Attention]] 
[[RCDA]] Row-column Decoupled Attention


---


- Additive Attention (Bahdanau 2014) : Au lieu de $QK^T$, utilise une couche MLP : $\text{score}(q_i,k_j) = v^T\tanh(W_q q_i + W_k k_j)$. Plus expressive mais plus lente (pas matricielle). 

- Linear Attention (Katharopoulos 2020) : Remplace softmax par kernel : $\phi(q)^T\phi(k)$ puis normalise. Complexité O(n) au lieu de $O(n^2)$, parfait pour longues séquences.

- Performer (Choromanski 2021) : Approximation FAVOR+ avec kernels aléatoires positifs, quasi-linéaire et sans perte de performance.

- Linformer : Projection low-rank de $Q$ et $K$ pour réduire à $O(n)$.

Note perso : Résultats de recherche IA rapide, doit être approfondi et vérifier si utilité.






