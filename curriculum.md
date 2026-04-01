# Training Curriculum: Pre-Training and RL Initialization

To prevent the agent from collapsing during the high-variance initial stages of Reinforcement Learning, the networks undergo a three-phase curriculum. This phases the agent from supervised Behavioral Cloning into Value Bootstrapping, and finally into True RL.

## Phase 1: Behavioral Cloning (Actor Pre-Training)

**Objective:** Jumpstart the shared trunk and Actor heads by mimicking hardcoded, rule-based heuristics. This maps the fundamental arithmetic of the game into the embedding space before gradients are subjected to the noise of RL.

### 1.1 Teacher Heuristics (The Dataset)
Generate an offline dataset of game states by simulating matches using algorithmic bots:
* **Match Teacher (Greedy-Match):** Always selects the valid Hand-Pool combination that yields the highest immediate point value.
* **Play Teacher (Stingy-Play):** Always discards the card with the lowest point value or lowest probability of being matched by the opponent.

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
* **Rollout:** Execute thousands of environment steps where the Agent uses its Phase 1 pre-trained policy to play against the Dummy Opponent (also using the Greedy/Stingy logic).

### 2.2 Critic Optimization
* Record the true Net Point Gap ($R_{step}$) for each turn.
* Feed the God-Mode sequences ($S_{pre\_match}$ and $S_{post\_play}$) into their respective Critic networks.
* **Loss Function:** Optimize both independent Critic branches using Mean Squared Error (MSE) against the empirical returns:
  $$Loss_{Critic} = MSE(Critic_{Agent}(S_{pre\_match}), R_{step}) + MSE(Critic_{Opponent}(S_{post\_play}), R_{step})$$

---

## Phase 3: True Reinforcement Learning (PPO)

**Objective:** Unfreeze the network and transition to self-directed policy optimization to surpass the limitations of the hardcoded teacher heuristics.

### 3.1 Unfreeze and Initialize
* Unfreeze all parameters (Embeddings, Shared Layer 1, Actor Branches, Critic Branches).
* Initialize the PPO Rollout Buffer.

### 3.2 The PPO Loop
* Begin collecting trajectories using the Turn-Based Net-Gap Macro-Step.
* Because the Actor already knows the basic rules (from Phase 1) and the Critic already understands the baseline point-gap expectations (from Phase 2), the calculated Advantages ($A = R - V$) will carry pure, high-quality strategic signals from Step 1 of the RL loop.
* The Agent will naturally begin to discover bluffing, trap-setting, and risk-management strategies that the Greedy/Stingy teachers were incapable of executing.
