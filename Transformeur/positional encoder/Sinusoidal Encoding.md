c'est un [[Positional Encoder]]
formule :
$\Large PE_{(pos,2i)} = \sin\left(\frac{pos}{10000^{2i/(d_{model})}}\right)$ 
$\Large PE_{(pos,2i+1)} = \cos\left(\frac{pos}{10000^{2i/(d_{model)}}}\right)$
avec :
- _pos_ = position du mot dans la phrase
- _d_model_ = dimension du modèle
- i = dimension index (de $0$  a $d_{model}/2$)
- _10.000 (n)_ = scaling factor control la variation de fréquence (constante déterminée expérimentalement)
## Exemples Sinusoidal Encoding
### Exemple 1D (texte)
**Paramètres :**
- `d_model = 512` (Attention is All You Need)
- `pos = 3`
- Premières dimensions seulement
**Calcul pour `i = 0` :**
$PE(pos=3,2i=0) = sin(3/10000^0) = sin(3) ≈ 0.14$ 
$PE(pos=3,2i+1=1) = cos(3) ≈ -0.99$

**Calcul pour `i = 1` :**
$2/512 = 0.0039$
$10000^0.0039 ≈ 1.035$
$3/1.035 ≈ 2.90$
$PE(pos=3,2i=2) = sin(2.90) ≈ 0.24$
$PE(pos=3,2i+1=3) = cos(2.90) ≈ -0.97$

**Vecteur résultat :**
[0.14, -0.99, 0.24, -0.97, ...]

**Intuition :**
- Dimensions basses → changements rapides (relations locales)
- Dimensions hautes → changements lents (relations globales)
---
### Exemple 2D (image)
**Paramètres :**
- `d_model = 512` → 256 dim. pour x, 256 pour y
- Position pixel : `(x=2, y=5)`

**Encodage x (pos = 2)**
$PE_x = [sin(2), cos(2), sin(2/1.035), cos(2/1.035), ...]$ 
$≈ [0.91, -0.42, 0.88, -0.47, ...]$

**Encodage y (pos = 5)**
$PE_y = [sin(5), cos(5), sin(5/1.035), cos(5/1.035), ...]$
$≈ [-0.96, 0.28, -0.98, 0.20, ...]$

**Vecteur final (concaténation) :**
$PE_(x=2,y=5) = [PE_x, PE_y]$

### ## Heatmap of positional encodings for a sentence with 50 words :
![[Pasted image 20260429112401.webp|697]]
The graph visualizes positional encodings for a sentence with 50 words, each word represented by a 128-dimensional vector. The vectors are displayed using a heatmap where colors represent different values.

## Intégration avec les embeddings
Le positional encoding est **ajouté** au vecteur d’embedding :
- embedding final = embedding + positional encoding

Cela injecte l’information de position directement dans la représentation.
**Why Addition Over Concatenation?**
The authors of the “Attention Is All You Need” paper opted for addition instead of concatenation. Concatenating would double the dimensionality of the vectors (from 6 to 12 in this case), leading to an increase in the number of parameters and, consequently, the training time. Addition keeps the conditionality consistent and avoids this increase, making the training process more efficient.
## Critique
- Ajout direct aux embeddings ce qui pollue l'information présente.
-  Encode une **position absolue** et pas relative. Pas possible de connaître la distance relative entre 2 [[Embeddings]]. Mais deux  [[Embeddings]] proches ont plus de chances d'être reliés.
