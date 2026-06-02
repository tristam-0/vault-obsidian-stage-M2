
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
le but est sa stabiliser la variance des valeur utiliser.
![[test.webp]]