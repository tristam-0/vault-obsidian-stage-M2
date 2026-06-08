## Liste des 4 configurations à tester

### 1. Baseline (Le modèle d'origine)

- **Configuration :** Sans enlever le Softmax.
- **Objectif :** Servir de point de comparaison (la référence) pour toutes les autres versions en termes de courbes de perte (Loss) et de précision (mAP).
    ### 2. DCNv4 Linéaire Pure
- **Configuration :** Juste enlever le Softmax (poids bruts non bornés).
- **Objectif :** Tester l'expressivité maximale du modèle, au risque de voir apparaître des valeurs extrêmes (comme le $-20$ dont tu parlais) qui peuvent faire exploser les gradients.

### 3. DCNv4 Sécurisé par ReLU
- **Configuration :** Enlever le Softmax et ajouter une activation **ReLU**.
- **Objectif :** Éviter les `NaN` et l'explosion des gradients. La ReLU va écraser à $0$ toutes les valeurs négatives (comme le faisait indirectement le Softmax) tout en permettant aux scores positifs de monter aussi haut qu'ils le veulent (expressivité).
### 4. DCNv4 Optimisé (Ton Idée)

- **Configuration :** Enlever le Softmax, ajouter la **ReLU**, et **forcer un biais initial positif faible** (ex: `0.1`) dans les paramètres.
- **Objectif :** Profiter de l'expressivité de la ReLU tout en garantissant qu'au tout début de l'entraînement, aucun point d'attention ne soit "mort" à $0$. Cela devrait donner la convergence la plus rapide et stable dès l'époque 1.