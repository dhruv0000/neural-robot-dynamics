### 1. Architectural Upgrade: Replace GPT-2 with Mamba (SSM)
The original paper uses a "lightweight GPT-2" (Transformer). While effective, Transformers have quadratic complexity with sequence length and can be computationally heavy for real-time simulation.
* **The Novelty:** Replace the causal Transformer with a **State Space Model (SSM)** like **Mamba** or **Mamba-2**.
* **Why it works:** SSMs offer **linear scaling** with sequence length and faster inference (recurrence) while maintaining the long-range dependency modeling of Transformers. This fits perfectly with the goal of a "fast, efficient neural simulator."
* **Hypothesis:** You could demonstrate that a Mamba-based NeRD is $2\times$ faster at inference time with equal or better stability than the Transformer baseline.

### 2. Parameter Efficiency: Mamba Layer Ablation (Mamba-3 vs Mamba-6)
To rigorously test the parameter efficiency of State Space Models, we implemented two distinct variants of the Mamba architecture:
* **Mamba-6:** A 6-layer configuration (~1.52M parameters), serving as the direct architectural counterpart to the baseline Transformer.
* **Mamba-3:** A lightweight 3-layer configuration (~761K parameters).
* **The Novelty:** By comparing `mamba-3` against `mamba-6` and the baseline, we aim to demonstrate that SSMs can achieve competitive dynamics modeling performance with **significantly fewer parameters** (approx. 50% reduction vs Mamba-6 and 72% reduction vs Transformer). This highlights the superior inductive bias of SSMs for continuous physics simulation.

### 3. Architectural Upgrade: Hybrid Jamba Architecture (Mamba + Transformer)
While Mamba offers efficiency, Transformers excel at in-context learning and high-fidelity recall. A pure Mamba model might struggle with complex, non-Markovian dynamics that a Transformer captures easily.
* **The Novelty:** Implement a **Hybrid Architecture (Jamba)** that interleaves Mamba blocks with Transformer (Attention) blocks.
* **Why it works:** This combines the best of both worlds: the **linear scaling and efficient state management** of Mamba with the **high-quality attention mechanism** of Transformers for critical dependency modeling.
* **Implementation:** We implemented a modular `JambaBlock` that stacks Mamba layers followed by a Transformer layer (e.g., Mamba -> Mamba -> Transformer). We optimized the layer count to ensure the total parameter count (~1.9M) remains comparable to the baseline Transformer (~2.7M) and pure Mamba (~1.5M) models, allowing for a fair comparison of architectural efficiency.

### 4. WandB Logging Metrics
The training pipeline logs comprehensive metrics to Weights & Biases (and TensorBoard) to track performance and stability.

**Training Metrics:**
* `params/lr/epoch`: Current learning rate.
* `training/train_loss/epoch`: Average training loss.
* `training_info/state_MSE/epoch`: Mean Squared Error of state predictions.
* `training_info/q_error_norm/epoch`: Error norm for joint positions.
* `training_info/qd_error_norm/epoch`: Error norm for joint velocities.

**Validation Metrics:**
* `training/valid_{dataset}_loss/epoch`: Validation loss for each dataset.
* `validating_info/state_MSE_{dataset}/epoch`: Validation MSE.

**Evaluation Metrics (Rollout):**
* `eval_{horizon}-steps/error(MSE)/epoch`: MSE over the full rollout horizon.
* `eval_{horizon}-steps/q_error(MSE)/epoch`: MSE for joint positions over rollout.
* `eval_details/error(MSE)_step_{i}/epoch`: Step-wise error accumulation.