**AMP** signifie **Automatic Mixed Precision** (Précision Mixte Automatique)
### 1. Le concept simple

Pour comprendre l'AMP, il faut savoir comment les GPU font leurs calculs :

- **FP32 (Single Precision - 32 bits) :** C'est la précision "standard". Elle est très précise, mais elle est lourde à calculer et prend beaucoup de place dans la mémoire du GPU.
    
- **FP16 (Half Precision - 16 bits) :** Elle utilise deux fois moins de mémoire et permet au GPU de faire les calculs beaucoup plus vite. **Le problème :** Elle est moins précise, ce qui peut parfois faire "dérailler" l'entraînement de ton modèle (les valeurs deviennent trop grandes ou trop petites).
    

**L'AMP**, c'est le "meilleur des deux mondes" : le système choisit intelligemment pour chaque calcul si la précision FP32 est nécessaire ou si la FP16 suffit

# he Mémo : AMP (Automatic Mixed Precision) sur les architectures DETR

> **En résumé :** L'AMP fait gagner énormément de temps et de mémoire VRAM, mais les calculs géométriques des boîtes englobantes dans DETR supportent très mal la basse précision du format FP16 standard, ce qui provoque régulièrement des plantages avec des pertes qui passent à `NaN` (Not a Number).

## 🚀 1. Les Bénéfices (Pourquoi l'activer ?)

- **Gain de vitesse massif :** Sur les GPU compatibles (avec _Tensor Cores_), l'entraînement peut être **1,5 à 3 fois plus rapide**.
    
- **Économie de mémoire (VRAM) :** Les activations et les poids du modèle prennent deux fois moins de place en mémoire.
    
- **Augmentation du Batch Size :** Moins de mémoire consommée signifie que tu peux doubler (voire tripler) ton `batch_size` sur un même GPU, ce qui stabilise l'entraînement de DETR.
    

## ⚠️ 2. Les Risques (Pourquoi ça peut crasher ?)

- **L'Underflow des boîtes (L1​ & GIoU) :** DETR régresse des coordonnées très précises. En FP16, les petits nombres ou les gradients minuscules s'arrondissent à zéro (underflow). Le modèle perd le fil de la géométrie et produit des `NaN`.
    
- **Le Softmax des têtes d'attention :** Dans le Transformer, les multiplications Q×KT en FP16 génèrent parfois des valeurs aberrantes (`-inf`). Après l'application du Softmax, ces lignes deviennent des `NaN` qui infectent tout le réseau.
    
- **L'instabilité d'AdamW :** L'epsilon par défaut d'AdamW (`1e-8`) est trop petit pour être lu en FP16 (le minimum étant `6e-5`). L'optimiseur fait alors des divisions par zéro en interne.
    

## 🛠️ 3. Les Solutions Matérielles

Le comportement de l'AMP dépend entièrement de la génération du GPU que tu utilises :

|Génération GPU|Modèles (exemples)|Comportement avec l'AMP|Recommandation|
|---|---|---|---|
|**Pascal (2016)**|NVIDIA P100|FP16 pur (sans Tensor Cores). Risque maximal de `NaN`. Aucun gain de vitesse.|❌ **Désactiver l'AMP** (Rester en FP32).|
|**Volta (2017)**|NVIDIA V100|FP16 pour le stockage, mais accumulation des calculs en FP32 grâce aux Tensor Cores.|⚠️ **Activer l'AMP (FP16)**, mais surveiller les pertes.|
|**Ampere+ (2020+)**|A100, H100, RTX 30/40|Support natif du **BF16** (Bfloat16), qui a la même dynamique que le FP32.|**Activer l'AMP (BF16)**. Zéro risque de `NaN`.|
https://github.com/facebookresearch/dino/issues/58

semple étre présent dans dino ? avoir plus tare