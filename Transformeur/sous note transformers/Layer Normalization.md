
## Pourquoi ?
Quand on entraîne un réseau de neurones, les caractéristiques d'entrée numériques peuvent avoir des échelles très différentes. Cela ralentit fortement la convergence du gradient descendant et peut causer des [[gradients explosifs ]] qui déstabilisent l'entraînement.


## fonctionnement
Let's consider an example where we have three vectors:
1. x1=[3.0,5.0,2.0,8.0]
2. x2=[1.0,3.0,5.0,8.0]
3. x3=[3.0,2.0,7.0,9.0]
For each input xx of the layer, Layer Normalization computes the following:

### Compute Mean and Variance for Each Feature
Mean and variance are calculated for each input but instead of across the batch, it’s done for the features (i.e per data point):
$$
\LARGE \mu = \frac{1}{H} \sum_{i=1}^{H} x_i
$$
$$
\LARGE \sigma^2 = \frac{1}{H} \sum_{i=1}^{H} (x_i - \mu)^2
$$

H is the number of features (neurons) in the layer
$x_i$​​  is the input for each feature
$μ$ and $σ^2$ are the computed mean and variance.
![[Pasted image 20260429122349.webp]]
Each feature is then normalized using the formula:
$\LARGE \hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}$
Here ϵ is a small constant added for numerical stability.
We calculate each normalized value
x1′​=[−0.6547,0.2182,−1.0911,1.5275]
x2′​=[−1.2568,−0.4834,0.2900,1.4501]
x3′​=[−0.7863,−1.1358,0.6116,1.3106]

To ensure that the normalized activations can still represent a wide range of values, learnable parameters γ (scaling) and β (shifting) are introduced. Final output y is computed as:

$y_i=γ\hat{x}_i +β$

This allows the network to scale and shift the normalized activations during training.
For simplicity, the following example uses scalar γγ and ββ only to show the calculation steps. Here let’s assume γ=1.5 and β=0.55. We can apply this scaling and shifting to the normalized values to get the final output for each vector.

- **For**  x1: y1=[−0.4820,0.8273,−1.1366,2.7913]
- **For** x2​: y2=[−1.3851,−0.2250,0.9350,2.6751]
- **For** x3​: y3=[−0.6795,−1.2037,1.4174,2.4658]

voir : https://docs.pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
