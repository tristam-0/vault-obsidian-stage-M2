Cet article propose des modification de architecture [[DETR]]
## Amélioration par rapport à DETR. 
amélioration des performances sur les petits objets et meilleures performances obtenues en 10 fois moins d'entraînement 

## Multi scale feature map 
Contrairement à [[DETR]], ou on récupère uniquement la dernière couche du CNN.
Ici on va récupérer les trois dernières couches et s'en servir pour générer quatre feature map à l'aide d'opérations de convolution voire schéma ci-dessous.
![[Pasted image 20260505104530.png]]
ensuite comme dans DETR Les feature map est aplatie en des séquence de taille d×HW.