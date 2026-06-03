
# 🔍 Micro-différences d'Architecture : DETR vs Anchor-DETR

Cette note récapitule les modifications  apportées par les auteurs d'Anchor-DETR (et Deformable DETR) sur les têtes de projection juste après le Backbone.

---

## 1. Ajout d'une couche de Normalisation (`GroupNorm`)

### 📍 Constat dans le code
Dans DETR standard, la projection des caractéristiques du backbone se fait via une convolution $1 \times 1$ pure (`nn.Conv2d`). Dans Anchor-DETR, cette convolution est immédiatement suivie d'un `nn.GroupNorm(32, hidden_dim)`.
### Hypothèses pourquoi ?
* **Stabilité du signal pour l'attention :** Le mécanisme d'attention (produits scalaires Query-Key) est extrêmement sensible à la variance des données d'entrée. Le `GroupNorm` force les caractéristiques visuelles à adopter une distribution stable (centrée en 0, variance contrôlée), empêchant les scores d'attention de saturer.
* **Indépendance de la taille du batch :** Contrairement au `BatchNorm`, le `GroupNorm` calcule ses statistiques par groupes de canaux et non par batch. C'est utile en détection d'objets où la taille de batch est souvent très petite (ex: 2 ou 4 images par GPU) à cause de la taille des images.
---
## 2. Initialisation forcée (Xavier Uniform & Biais à 0)

### 📍 Constat dans le code
À la fin de l'initialisation des projections, le code applique  :
```python
for proj in self.input_proj:
    nn.init.xavier_uniform_(proj[0].weight, gain=1)
    nn.init.constant_(proj[0].bias, 0)
```
Seule la convolution (`proj[0]`) est ciblée, la normalisation est ignorée.
le but est de stabiliser la variance entre les layer et qui lute contre Vanishing Gradient ou Exploding Gradient.
![[mico diferanceimg.webp|527]]
![[mico difference-1780390147512.webp]]
On constate que les modèles avec initialisation normalisée (notés N) sont plus rapides à l'entraînement et donnent de meilleurs résultats (jusqu'à 11 % de gain sur des tâches visuelles complexes, et plus généralement de 1 à 3 % d'amélioration sur les autres jeux de données de l'article).

pour plus d'information
https://www.youtube.com/watch?v=lyN-OCCrhuo&pp=ygUcd2VpZ2h0IGluaXRpYWxpemF0aW9uIHhhdmllcg%3D%3D
https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

---

autre arg dif
code originale
parser.add_argument('--dim_feedforward', default=2048, type=int,
help="Intermediate size of the feedforward layers in the transformer blocks")
code ancode
parser.add_argument('--dim_feedforward', default=1024, type=int,
help="Intermediate size of the feedforward layers in the transformer blocks")

passage de 2028 -> 1024

---
dropout
passage de 0.1 a 0