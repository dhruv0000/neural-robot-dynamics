### 1. Architectural Upgrade: Replace GPT-2 with Mamba (SSM)
The original paper uses a "lightweight GPT-2" (Transformer). While effective, Transformers have quadratic complexity with sequence length and can be computationally heavy for real-time simulation.
* **The Novelty:** Replace the causal Transformer with a **State Space Model (SSM)** like **Mamba** or **Mamba-2**.
* **Why it works:** SSMs offer **linear scaling** with sequence length and faster inference (recurrence) while maintaining the long-range dependency modeling of Transformers. This fits perfectly with the goal of a "fast, efficient neural simulator."
* **Hypothesis:** You could demonstrate that a Mamba-based NeRD is $2\times$ faster at inference time with equal or better stability than the Transformer baseline.

### 2. Training Stability: Implement "Temporal Unrolling" (Push-forward Loss)
The paper primarily uses "teacher forcing" (predicting $t+1$ given ground truth at $t$). This can lead to "exposure bias" where the model drifts when running on its own predictions.
* **The Novelty:** Implement **multi-step loss** (or "temporal unrolling") during training. Instead of just predicting $t+1$, force the model to predict $t+1 \dots t+10$ autoregressively during training and backpropagate through the entire chain.
* **Why it works:** This directly trains the model to recover from its own small errors, significantly improving the "stability horizon" (e.g., from 1,000 steps to 5,000 steps).

```
Novelty Features Walkthrough
I have implemented two major novelty features to the Neural Robot Dynamics codebase:

Mamba Architecture: A State Space Model (SSM) alternative to the Transformer.
Temporal Unrolling: A training stability technique using autoregressive loss.
Changes
1. Mamba Architecture
I added a pure PyTorch implementation of the Mamba (S6) block in 
models/mamba.py
. This allows experimenting with SSMs without needing complex CUDA kernels (though it will be slower than optimized versions).

> [!NOTE]
> **Implementation Detail:** Unlike the standard Mamba implementation which often assumes inputs are already embedded, this implementation includes an internal linear projection layer (`self.embedding`) to map the input feature dimension (e.g., state size) to the model's hidden dimension (`d_model`). This mirrors the behavior of the GPT implementation's `wte` layer, ensuring compatibility with the rest of the training pipeline.

File: 
models/mamba.py
Usage: Use --novelty mamba in the training CLI.
2. Temporal Unrolling
I modified the 
VanillaTrainer
 to support an 'unroll' mode. This mode forces the model to predict a sequence of states autoregressively during training, calculating loss on the entire generated trajectory.

File: 
algorithms/vanilla_trainer.py
Usage: Use --novelty unroll in the training CLI.
3. CLI Updates
I updated 
train/train.py
 to accept new arguments:

--novelty: Choose between mamba or 
unroll
.
--sample-sequence-length: Override the sequence length defined in the config.
Verification
To verify the changes, you can run the following commands (ensure dependencies are installed):

Verify Mamba
python train/train.py --cfg ./train/cfg/Ant/transformer.yaml --novelty mamba --logdir ./data/test_mamba --num_epochs 1 --num_iters_per_epoch 10
Verify Unrolling
python train/train.py --cfg ./train/cfg/Ant/transformer.yaml --novelty unroll --logdir ./data/test_unroll --num_epochs 1 --num_iters_per_epoch 10
Verify Custom Sequence Length
python train/train.py --cfg ./train/cfg/Ant/transformer.yaml --sample-sequence-length 20 --logdir ./data/test_seq_len --num_epochs 1 --num_iters_per_epoch 10
NOTE

Verification requires a full installation of NVIDIA Warp (warp-lang) and its dependencies (e.g., warp.sim). In some environments, this may require manual installation steps not covered by 
requirements.txt
.