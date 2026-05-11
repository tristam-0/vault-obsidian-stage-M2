
### Entraînement
- L’entrée est la séquence cible décalée vers la droite avec un token special de début (`<BOS>`)
- On donne au modèle les **vrais tokens** à chaque étape (teacher forcing)
- Exemple :  
    entrée → `<BOS> La pizza est`  
    cible → `La pizza est délicieuse`
### Inférence
- L’entrée est constituée des **tokens déjà générés** par le modèle
- La génération se fait **pas à pas**, en réinjectant chaque nouveau token
- Exemple :  
    étape 1 → `<BOS>` → prédit "La"  
    étape 2 → `<BOS> La` → prédit "pizza"  
    étape 3 → `<BOS> La pizza` → prédit "est"
    étape 4 → `<BOS> La pizza est` → prédit "délicieuse"