# Training Curriculum: Pre-Training and RL Initialization

To prevent the agent from collapsing during the high-variance initial stages of Reinforcement Learning, the networks undergo a three-phase curriculum. This phases the agent from supervised Behavioral Cloning into Value Bootstrapping, and finally into True RL.

## Phase 1: Behavioral Cloning (Actor Pre-Training)

**Objective:** Jumpstart the shared trunk and Actor heads by mimicking hardcoded, rule-based heuristics. This maps the fundamental arithmetic of the game into the embedding space before gradients are subjected to the noise of RL.

### 1.1 Teacher Heuristics (The Dataset)
Generate an offline dataset of game states by simulating matches using algorithmic bots aligned with **rl.md §1.5 baseline**:
* **Match Teacher (Greedy-for-public):** Prioritizes the public pool card’s suit/joker point, then total capture points, then hand-card count (see `GreedyForPublicMatchPolicy`).
* **Play Teacher (Caution-Play):** Prefers discarding by largest game digit first, then lower point value among ties (`CautionPlayPolicy`).

### 1.2 Recursive/Balanced Trunk Training
Because the Match Head and Play Head share the Token Embeddings and Transformer Layer 1, they cannot be trained sequentially (e.g., training all Match data, then all Play data), as this will cause catastrophic forgetting in the shared weights. 
* **Method:** Training must be recursively balanced. 
* **Implementation:** During each forward pass, sample a mixed batch containing both Match states and Play states.
* **Loss Function:** Compute the Cross-Entropy (CE) loss for both heads simultaneously and backpropagate the joint loss to stabilize the shared layer:
  $$Loss_{BC} = CE(Actor_{match}, Teacher_{match}) + CE(Actor_{play}, Teacher_{play})$$

---

## Phase 2: Value Bootstrapping (Critic Pre-Training)

**Objective:** Calibrate the two God-Mode Critic networks to accurately predict the point gap based on the newly established baseline policy, ensuring PPO begins with stable, grounded Advantage calculations.

### 2.1 Policy Freeze and Rollout
* **Action:** Freeze the weights of the Token Embeddings, Shared Layer 1, and both Actor Layer 2s.
* **Rollout:** Execute thousands of environment steps where the Agent uses its Phase 1 pre-trained policy to play against the Dummy Opponent (same **baseline** bundle: greedy-for-public match + caution play).

### 2.2 Critic Optimization
* Record the true Net Point Gap ($R_{step}$) for each turn.
* Feed the God-Mode sequences ($S_{pre\_match}$ and $S_{post\_play}$) into their respective Critic networks.
* **Loss Function:** Optimize both independent Critic branches using Mean Squared Error (MSE) against the empirical returns:
  $$Loss_{Critic} = MSE(Critic_{Agent}(S_{pre\_match}), R_{step}) + MSE(Critic_{Opponent}(S_{post\_play}), R_{step})$$

### 2.3 Maintaining Actor Accuracy (Recursive / Interleaved Schedule)
Actor weights (embeddings, shared Layer 1, match/play Layer 2s) are **frozen** during pure critic updates so Phase 1 BC accuracy (match, play, and overall) is **preserved by construction**. The reference implementation may optionally **interleave** short BC refresh epochs (critics frozen, joint match+play CE on a cached teacher dataset) between critic blocks if shared layers are ever trained jointly; with a strict freeze, verify BC metrics on the held-out split after Phase 2 match Phase 1 targets (e.g. well above 95% overall).

---

## Phase 3: True Reinforcement Learning (PPO)

**Objective:** Unfreeze the network and transition to self-directed policy optimization to surpass the limitations of the hardcoded teacher heuristics.

### 3.1 Unfreeze and Initialize
* Unfreeze all parameters (Embeddings, Shared Layer 1, Actor Branches, Critic Branches).
* Initialize the PPO Rollout Buffer.

### 3.2 The PPO Loop
* Begin collecting trajectories using the Turn-Based Net-Gap Macro-Step.
* Because the Actor already knows the basic rules (from Phase 1) and the Critic already understands the baseline point-gap expectations (from Phase 2), the calculated Advantages ($A = R - V$) will carry pure, high-quality strategic signals from Step 1 of the RL loop.
* The Agent will naturally begin to discover bluffing, trap-setting, and risk-management strategies beyond the baseline heuristic teacher.
# Phase 4: Reinforcement Learning Telemetry and Monitoring

**Objective:** Establish a robust monitoring framework to track environment performance and mathematical stability as the agent transitions from static pre-training to high-variance, self-directed Proximal Policy Optimization (PPO).

## 4.1 Environment Metrics (Game Performance)
These metrics evaluate the agent's actual gameplay capability and strategic development.

* **Average Episode Return (Net Point Gap):** The primary optimization target. Tracks the average of `Agent Score - Opponent Score` per episode. Must show an upward trend into the positive range.
* **Win Rate:** A lagging indicator of success. Useful for human readability, but less sensitive to incremental improvements than the Net Point Gap (e.g., losing by a smaller margin is mathematically tracked as improvement, but does not increase win rate).
* **Phase-Specific Reward Split:** Tracks the average Match Reward and average Play Reward independently. Ensures the agent is balancing offensive gains (Match) with defensive risk management (Play) rather than collapsing into a one-sided strategy.

## 4.2 PPO Health Metrics (Network Stability)
These metrics diagnose the mathematical health of the gradient updates and prevent policy collapse due to environment variance.

* **Policy Entropy:** Measures the Actor's uncertainty and exploration rate. 
  * *Target:* Starts high during initial exploration and smoothly decays as confidence grows.
  * *Failure State:* A sudden crash to `0.0` indicates "Policy Collapse" (the agent became overly penalized by variance, locked onto a single safe action, and stopped exploring).
* **Approximate KL Divergence ($D_{KL}$):** Measures the magnitude of the policy distribution change per gradient update.
  * *Target:* Remains stable, typically oscillating between `0.005` and `0.02`.
  * *Failure State:* Wild spikes ($>0.1$) indicate the learning rate is too high or the rollout batch size is too small to mathematically absorb the randomness of the card draws.
* **Critic Explained Variance ($EV$):** Measures the Critic's predictive accuracy against the empirical returns.
  * *Target:* Plateaus in the positive range (typically $>0.1$ for hidden-information card games).
  * *Failure State:* Dips below `0.0`, indicating the Critic's predictions are worse than guessing the average score, which will actively corrupt the Actor's Advantage calculations.
* **Value Loss (Critic Loss):** The Mean Squared Error of the `Critic-Agent` and `Critic-Opponent` predictions. Expected to spike initially during exploration of new states and subsequently plateau.

## 4.3 Telemetry Infrastructure
To visualize the high-frequency data streams generated by the PPO loop, a dedicated logging framework is integrated directly into the training script.

* **Primary Integration: Weights & Biases (W&B / `wandb`)**
  * *Implementation:* Cloud-based visualization initialized via `wandb.init(project="card-agent-ppo")`.
  * *Advantage:* Enables real-time remote monitoring, automatic gradient tracking, and seamless overlaying of historical runs for hyperparameter ablation studies.
* **Alternative Integration: TensorBoard**
  * *Implementation:* Local logging via PyTorch's `torch.utils.tensorboard.SummaryWriter`.
  * *Advantage:* Completely offline and open-source, suitable for air-gapped or entirely localized training environments.
