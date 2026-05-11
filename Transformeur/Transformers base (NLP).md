We can see that it consists of [[Encoders]] and [[Decoders]]

![[Pasted image 20260427111205.webp]]

## 1. Embedding

Avant d'être passée à l'encodeur, la phrase « I like Pizza » est décomposée en ses mots respectifs et chaque mot est encodé à l'aide d'une matrice [[Embeddings]] qu'on entraîne.
Puis on ajoute [[Positional Encoder]] pour obtenir un [[Embeddings]] positionnelle.
![[Pasted image 20260427112717.webp]]
### Exemple positional encoding
![[Sinusoidal Encoding]]
## 2. Encoder
Après les [[Embeddings]], on passe au bloc [[Encoders]].

![[Encoders]]

## 3. Decoder
Une fois qu’une représentation est passée par les blocs  [[Encoders]], on utilise cette représentation (ses features) dans la couche de cross‑attention du décodeur.

![[Decoders]]