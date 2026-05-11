![[Pasted image 20260429155655.webp]]

 illustration d'un transformer vision encodeur stack uniquement 

## patch embedding
Différentes méthodes sont possibles pour traiter des images. 
Par exemple, dans article 'An Image is Worth 16x16 Words'

l'embedding est réalisé de la manière suivante. 
Pour réaliser l’embedding des différents patchs, on divise l’image en patchs de taille fixe (par exemple 16×16 pixels) puis on les projette linéairement à l’aide d’une opération de convolution 2D.
Le modèle étant **invariant par permutation** des positions, il faut rajouter une information de position spatiale.
- On utilise un **embedding positionnel** (positional encoding).
- Pour un patch situé à la position (i,j) dans la grille, on peut, par exemple, utiliser un [[Sinusoidal Encoding]] 2D, où :
    - une moitié du vecteur encode la coordonnée x=i (indice de ligne),
    - l’autre moitié encode la coordonnée y=j (indice de colonne).
    - Les deux sont concaténés (ou combinés) puis **ajoutés** au patch embedding correspondant
##  Encodeur décodeur

Contrairement aux modèles NLP classiques qui utilisent souvent un **encodeur–décodeur**, un Vision Transformer typique utilise **uniquement un stack d’encodeurs**.
- Chaque couche d’encodeur applique :
    - **Multi‑head self‑attention** sur la séquence de tokens de patchs,
    - puis un **feed‑forward network (FFN)** avec normalisation et résidus.
- L’architecture est donc très proche de celle d’un Transformer NLP : seule la manière de construire les tokens (patch embedding) change.