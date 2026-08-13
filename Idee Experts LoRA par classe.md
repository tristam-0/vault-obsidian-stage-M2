
# Idée : Experts LoRA par classe pour la tête de classification DETR (UDA teacher–student)

> **En résumé :** dans un DETR, utiliser la prédiction de classe de l'**avant-dernière**
> couche du décodeur comme *hypothèse de classe* par requête. Pour chaque classe Y,
> on ajoute l'équivalent d'un petit **expert LoRA dédié** sur la tête de classification
> finale (reste globale pour toutes les requêtes). Le modèle peut ainsi "poser la question
> *est-ce bien la classe Y ?*" et **rejeter** l'hypothèse, surtout quand la tête globale
> donne une probabilité faible. Évite de devoir swapper tous les poids (trop coûteux) :
> LoRA = adaptateur peu coûteux.

## L'idée en bref
1. Le décodeur DETR = encodeur + décodeur multi-couches.
2. L'avant-dernière couche du décodeur prédit une classe (PRN/FFN) ;
   c'est l'**hypothèse de classe**.
3. Pour chaque requête prédite de classe Y, on modifie les poids de la
   **dernière couche** : poids de base + **LoRA dédié à la classe Y**.
4. Résultat : la dernière couche peut **accepter ou rejeter** une classe
   avec une spécialisation ultra-fine par classe.

## Pourquoi LoRA et pas un swap complet
- Swap de tous les poids = trop coûteux.
- LoRA = petite adaptation (matrices low-rank), coût acceptable.
- Les LoRA agissent comme des **experts mono-classe**.

## Contexte visé
- Domain adaptation, surtout **teacher–student** (pseudo-labels).
- Le modèle émet une hypothèse de classe → peut la **rejeter** si la tête globale (commune à toutes les requêtes) donne une faible proba.
- Cible : réduire les faux positifs (ex. sous brouillard).

## Sanity check (exploration littérature)
- **Déjà fait ainsi ?** Non trouvé tel quel (avant-dernière couche → LoRA
  par classe sur la tête finale d'un DETR, routé par requête).
- **Briques validées :**
  - Vérification/refus en 2 étapes : **DCR** (Decoupled Classification Refinement),
    RefineDet (negative anchor filtering).
  - Routage conditionnel : MoE, confidence-gated routing,
    LoRA routing (LLM serving).
  - LoRA/experts par classe : **Polymorph** (LoRA par groupe de labels),
    RIDE (experts + routage), OV-DETR/OVLW-DETR (poids de classifieur par classe).
  - Plus proche : **HI-MoE** (routage par requête dans un DETR, mais routage
    appris, pas basé sur la classe prédite).