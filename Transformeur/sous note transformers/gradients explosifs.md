Les **gradients explosifs** surviennent quand les **valeurs d'entrée très grandes** se propagent à travers les couches du réseau via les **multiplications matricielles** (poids × entrées).

## exemple

on a un réseaux a 3 couche avec 1 neurone par couche on int tous a 0,7.

Entrée x₀ = 50 000
Couche 1 : z₁ = W₁ × x₀ = 0.7 × 50 000 = 35 000
ReLU(z₁) = 35 000
Couche 2 : z₂ = W₂ × 35 000 = 0.7 × 35 000 = 24 500
ReLU(z₂) = 24 500  
Couche 3 : z₃ = 0.7 × 24 500 = 17 150

## Règle de la chaîne en backpropagation

La formule **∇z₃/∇W₁** calcule comment un **petit changement** du poids W₁ (couche 1) affecte z₃ (sortie couche 3). C'est la **dérivée totale** via la **chaîne multiplicative** :

1. ∂z₁/∂W₁ = x₀ = 50 000
   (z₁ change de x₀ si W₁ change de 1)

2. ∂z₂/∂z₁ = W₂ = 0.7  
   (z₂ change de W₂ si z₁ change de 1)

3. ∂z₃/∂z₂ = W₃ = 0.7
   (z₃ change de W₃ si z₂ change de 1)

4. CHAÎNE COMPLÈTE :
   ∂z₃/∂W₁ = ∂z₃/∂z₂ × ∂z₂/∂z₁ × ∂z₁/∂W₁
           = 0.7 × 0.7 × 50 000
solution normalisation pour évité devoir 50 000 a l'entais du réseau