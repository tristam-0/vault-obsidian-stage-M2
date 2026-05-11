---
aliases:
  - Encodeur-Decoder
  - Encoder-Decoder
  - Cross-Attention
  - Cross attention
---
![[Pasted image 20260428141437.webp]]
## Self-Attention Decoders
le fonctionnement de la self-Attention dans le Decoders est  similaire a celui dans  l'encodeur voir note [[Self-Attention "Attention is all you need" architectures]] pour plus de détail.
La différence principale est l’ajout d’un **masque** pour empêcher chaque position de regarder les tokens futurs.
- Lors de la génération, le token en position $i$ ne peut utiliser que les tokens $≤i$
- Cela garantit une génération **auto-régressive** (on ne triche pas avec le futur)
Concrètement :
- On applique un masque triangulaire sur la matrice des scores d’attention
- Les positions interdites sont mises à −∞ avant le softmax
- Après softmax, ces positions ont un poids nul
Cela force le modèle à apprendre à prédire séquentiellement.
![[Pasted image 20260429092715.webp]]
## Cross attention
La cross-attention permet au décodeur de **se connecter à l’encodeur**.
- Les **Keys (K)** et **Values (V)** viennent de la sortie du dernier encodeur.
- Les **Queries (Q)** viennent de la sortie de la self-attention du décodeur
Le mécanisme fonctionne comme suit :
1. Chaque position du décodeur produit une **Query** $q_i$
2. Cette Query est comparée à toutes les **Keys** de l’encodeur
3. On obtient des scores d’attention qui indiquent quelles parties de l’entrée sont pertinentes.
4. Ces scores pondèrent les Values de l’encodeur
5. On obtient un vecteur contextualisé pour chaque position du décodeur
### Exemple simple
Traduction :  
Entrée (encodeur) : _"The pizza is delicious"_  
Sortie (décodeur) : _"La pizza est délicieuse"_

Quand le modèle génère **"délicieuse"** :
- La self-attention regarde _"La pizza est"_
- La cross-attention va fortement se concentrer sur _"delicious"_ dans l’entrée
Cela permet d’aligner correctement les mots entre les deux langues.