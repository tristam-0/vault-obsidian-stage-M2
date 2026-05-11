---
aliases:
  - Positional Encoding
---

## Quelle est l'utilité
Le mécanisme de self-attention ne prend pas en compte l’ordre des mots.  
Sans information de position, une phrase comme :
- « Pramod aime manger de la pizza avec ses amis »
- « Pramod aime manger des amis avec sa pizza »
- ou toute autre permutation des mots.
produirait la **même représentation**, car le modèle traite les tokens comme un ensemble sans ordre.

Le **positional encoding** permet donc d’ajouter explicitement l’information de position des mots dans la séquence, afin que le modèle puisse comprendre leur organisation et leur sens.
## Caractéristiques attendues
Un bon encodage positionnel doit respecter plusieurs propriétés importantes :
- **Encodage unique pour chaque position:** Parce que sinon il continuera à changer pour différentes longueurs de phrases. La position 2 pour une phrase de 10 mots sera différente de celle d'une phrase de 100 mots. Cela entravera la formation, car il n'y a pas de schéma prévisible qui puisse être suivi.
- **Relation linéaire entre deux positions codées**: Si je connais la position p d'un mot, il devrait être facile de calculer la position p + k d'un autre mot. Cela permettra au modèle d'apprendre plus facilement le motif.
- **Capacité de généralisation** : le modèle doit pouvoir traiter des séquences plus longues que celles vues pendant l’entraînement
- **Génération déterministe** : idéalement basé sur une formule simple (pas appris), pour éviter d’introduire trop de paramètres et améliorer la robustesse.
- **Extensibilité** : doit fonctionner pour différentes dimensions (utile en NLP mais aussi en vision ou autres variantes).
* **Échelle contrôlée** : les valeurs ne doivent pas dominer celles des embeddings, pour éviter d’écraser l’information sémantique.
## Encoder list

[[Sinusoidal Encoding]] : introduit dans _Attention is All You Need_, basé sur des fonctions sinus et cosinus
[[Rotary Position Encoding  (RoPE)]] : encode la position via une rotation dans l’espace des embeddings, très utilisé dans les modèles récents