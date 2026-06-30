improtent de reconert des ovject non vue comme "inconu"


olving World Object Detection (EWOD),

First, Incremental LoRA Adapters (Section 3.3) employ a dual-adapter architecture: an aggregate adapter that accumulates compressed knowledge from all previous tasks and a task-specific adapter that captures current task updates. Through data-aware merging guided by pertask sample ratios and low-rank projection via truncated SVD, we achieve stable knowledge consolidation without storing any exemplars while explicitly addressing data imbalance. Second, Query-Norm Objectness Adapter (Section 3.4) decouples query semantics from magnitude by normalizing decoder features, yielding domain-invariant classagnostic representations that enable robust unknown detection even under severe domain shifts—crucially, without requiring any auxiliary supervision or additional losses, unlike [7, 36]. Third, Entropy-Aware Unknown Mixing (Section 3.5) calibrates unknown predictions by combining classification uncertainty with objectness evidence, ensuring that high-objectness, high-uncertainty queries are correctly identified as unknowns rather than spuriously absorbed into known classes or background.


![[EW-DETR-figure1.webp]]

but de EWOD

(i) detect all known classes Kt across all seen domains {D1, . . . , Dt}; (ii) detect all unseen objects as “unknown” without explicit unknown supervision; (iii) incrementally learn a subset of unknowns as knowns when their labels are revealed in later tasks
(iv) achieve these objectives in an exemplar-free manner without storing any previous data



3 modulles intruduis
![[EW-DETR-figure3.webp]]
# Incremental LoRA adapters
Pour éviter l'oubli catastrophique (catastrophic forgetting) lors de l'apprentissage de nouvelles tâches, le modèle attache des **Incremental Low-Rank Adaptation ([[LoRA]]) adapters** aux FFN de l'encodeur et du décodeur du Transformer.


For each targeted layer at task t with frozen base weight W0, we maintain two low-rank adapters: (c'est quoi un low rabk adapters)
• Aggregate LoRA Adapter ∆Wt−1  agg : a non-trainable buffer that accumulates knowledge from all previous tasks {T1, . . . , Tt−1}, reused in subsequent tasks. 
• Task-Specific LoRA Adapter ∆Wt  task : Trainable parameters for the current task Tt to capture task-specific shifts in classes/domains, resets at each task transition.

∆Wt  task = Bt  taskAt  task, ∆Wt−1  agg = Bt−1  agg At−1  agg , (2)  During training at task t, only the task-specific LoRA matrices At  task and Bt  task (pourquoi 2 sous martrice ?)
## data ilmbalance
data-aware merging coefficient (βt), bounded by βmin and βmax, that is computed using the ratio between samples seen in current task (Nt) and the cumulative number of samples seen so far previously $(N_{1:t−1})$
$$\beta_t = \left\{ \begin{array}{ll} 1, & t = 1, \\[4pt] \beta_{\max} - (\beta_{\max} - \beta_{\min}) \frac{N_t}{N_{1:t-1}}, & t \geq 2. \end{array} \right.$$

pas sur de bien comprendre
∆Wt  merged = (1 − βt) ∆Wt−1  agg + βt ∆Wt  task. (4)  To keep the Aggregate LoRA adapter low-rank, we project this matrix back to rank r using a truncated Singular Value Decomposition (SVD): ∆Wt  merged = U Σ V⊤, and retain the top-r singular components, so that ∆Wt  merged ≈  Ur Σr Vr⊤. We store this low-rank approximation and use it to compute Aggregate LoRA adapter for Tt:  Bt  agg = Ur Σr, At  agg = V⊤  r , (5)  ∆Wt  agg = Bt  aggAt  agg (6)  While the Task-Specific LoRA adapter is reset


c'est quoi le mag il sor de ou ??


# Query-Norm Objectness Adapter

reparameterises the decoder output features to yield a decoupled representation that helps in “unknown” detection.
# entropy Aware Unknown Mixing module
both objectness and classification uncertainty to modulate the final class scores in a balanced manner.