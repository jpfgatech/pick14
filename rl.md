# Architecture Specification: Asymmetric PPO Agent for Zero-Sum Card Game

## 1. Game Structure and State Description

### 1.1 Game Flow
The game consists of sequential rounds containing alternating player turns. A single turn consists of the following phases:
1.  **Match:** Player selects hand card(s) and one public card to move to their scoring pile. A successful match requires the sum of the card digits to equal 14. If no match is available or desired, the player passes.
2.  **Draw 1:** If a match was made, the player draws cards until they have 4 cards in hand (or until the deck is exhausted).
3.  **Play:** If the player has 4 cards in hand or passed the match phase, they must discard 1 hand card to the public pool. Otherwise, this phase is skipped.
4.  **Draw 2:** If the player has fewer than 3 cards, they draw to refill to 3 (only possible if the match action was passed).

### 1.2 State Representation
The state is represented as a sequence of Card Sets (tokens). Each token embeds the card's physical identity and its ownership role.
* **Visible State (Agent):** Player Hand, Public Pool.
* **Hidden State (Critic):** Opponent Hand(s). (Deck and used cards are omitted).
* **Role Encoding:** A maximum of 4 type encodings are used: `Self`, `Pool`, and `Opponent`. In games with $>3$ players, all opponents except for the one acting after agent are aggregated under the `Opponent 2` encoding to represent the collective long-term threat. Initialize the Embedding as 8 LUT entries to support potential expansion, e.g. more players or deck/used cards.
* **Tokenization:** Each card set is converted into a vector containing zero-padded digits/points, total card count, and summed values. This is projected into a $32$-dimensional token. The Card token and Role token are added element-wise: $\text{Embedding} = \text{Card}_{32} + \text{Role}_{32}$.

---

## 2. Neural Network Architecture

The model utilizes a Tree-Structured Asymmetric Transformer architecture, operating on $32$-dimensional tokens.

### 2.1 Input Sequences
* **Agent Sequence:** Contains $M$ visible tokens (Hand + Pool) plus $32$ Global Context tokens (`[CLS]` or scratchpad variables). Total length: $M+32$.
* **Critic Sequence:** Contains $N$ total tokens (Hand + Pool + Opponent Hand) plus $32$ Global Context tokens. Total length: $N+32$.

### 2.2 Transformer Backbone
The network utilizes exactly 2 layers of Self-Attention to prevent overfitting while allowing hierarchical relational logic.
* **Shared Layer 1:** All network branches share the first Transformer layer to learn foundational card arithmetic and basic synergies.
* **Independent Layer 2:** The network splits into four independent, specialized Transformer layers: Match, Play, Critic-Agent, and Critic-Opponent.
* **Internal Operations (Per Layer):**
    1.  Multi-Head Self-Attention (MHA)
    2.  Residual Add & `LayerNorm`
    3.  Feed-Forward Network: `Linear(32, 128)` $\rightarrow$ `GELU` $\rightarrow$ `Linear(128, 32)`
    4.  Residual Add & `LayerNorm`

---

## 3. Task Heads

### 3.1 Match Head (Actor)
The Match Head outputs the probability distribution over valid Hand-Pool pairings using a Pointer Network mechanism built on top of the Transformer outputs.
1.  **Slice:** Extract the $H$ Hand tokens and $P$ Pool tokens from the $(M+32) \times 32$ output of the Match-specific Layer 2.
2.  **Project:** Pass Hand tokens through a Query projection and Pool tokens through a Key projection (`Linear(32, 32)`).
3.  **Cross-Score:** Compute the attention grid using dot products, scaled by the square root of the dimension to maintain gradient stability:
    $$\text{Logits} = \frac{Q_{hand} \times K_{pool}^T}{\sqrt{32}}$$
4.  **Output:** Apply a Legal Action Mask (setting invalid pairs to $-1e9$), flatten the $H \times P$ grid, and apply `Softmax(dim=-1)` to yield action probabilities.

### 3.2 Play Head (Actor)
The Play Head selects a single card to discard from the hand.
1.  **Slice:** Extract the $H$ Hand tokens from the $(M+32) \times 32$ output of the Play-specific Layer 2.
2.  **Project:** Apply a `Linear(32, 1)` layer to squeeze the embedding dimension, yielding $H$ raw scalars.
3.  **Output:** Apply a Legal Action Mask to invalid/empty slots (set to $-1e9$) and apply `Softmax(dim=-1)` to yield play probabilities.

### 3.3 Multi-Critic Heads (God-Mode)
To prevent gradient thrashing between the Agent's exploratory policy and the Opponent's baseline logic, two independent Critic networks evaluate the $(N+32) \times 32$ omniscient sequence.
1.  **Critic-Agent:** Evaluates $S_{pre\_match}$ to predict the point gap prior to the Agent's turn.
2.  **Critic-Opponent:** Evaluates $S_{post\_play}$ to predict the point gap prior to the Opponent's turn.
3.  **Funnel & Output (Both Critics):**
    * Average the $(N+32)$ tokens along the sequence dimension to produce a $1 \times 32$ summary vector (`Mean Pooling`).
    * Pass through a Multi-Layer Perceptron: `Linear(32, 16)` $\rightarrow$ `ReLU` $\rightarrow$ `Linear(16, 1)`.
    * No final activation is applied, allowing for continuous positive, zero, or negative point gap predictions.

---

## 4. Reward Logging and PPO Backpropagation

The environment uses a Turn-Based Net-Gap Macro-Step. The target metric is strictly the point gap between the Agent and the Opponent.

### 4.1 Atomic Reward Definitions
To calculate accurate discounted future returns without double-counting, the scoring events are decoupled into strictly sequential, alternating atomic rewards. 

Let $A_t$ be the points the Agent scores on turn $t$.
Let $O_t$ be the points the Opponent scores on their subsequent turn $t$.

* **Atomic Match Reward:** $r_{match}^{(t)} = A_t$
    *(The immediate offensive points gained by the Agent's match).*
* **Atomic Play Reward:** $r_{play}^{(t)} = -O_t$
    *(The immediate defensive penalty incurred by the opponent's response to the discard).*

### 4.2 Discounted Cumulative Returns ($G_t$)
The agent evaluates the quality of an action not just by its atomic reward, but by the infinite sum of discounted future rewards, governed by the discount factor $\gamma \in [0, 1)$.

**1. Match Action Return ($G_{match}$):**
The sequence from a Match action looks forward through the Play action and into the next turn.
$$G_{match}^{(t)} = r_{match}^{(t)} + \gamma r_{play}^{(t)} + \gamma^2 r_{match}^{(t+1)} + \gamma^3 r_{play}^{(t+1)} + \dots$$
*(Substituting the game metrics: $A_t - \gamma O_t + \gamma^2 A_{t+1} - \gamma^3 O_{t+1} \dots$)*

**2. Play Action Return ($G_{play}$):**
The sequence from a Play action looks forward into the opponent's immediate response and the Agent's subsequent offensive turn.
$$G_{play}^{(t)} = r_{play}^{(t)} + \gamma r_{match}^{(t+1)} + \gamma^2 r_{play}^{(t+1)} + \dots$$
*(Substituting the game metrics: $-O_t + \gamma A_{t+1} - \gamma^2 O_{t+1} \dots$)*

### 4.3 Dual Critics and Advantage Calculation
Because $G_{match}$ and $G_{play}$ evaluate staggered starting points of the exact same infinite sequence, the two Critic networks are optimized to predict these specific cumulative returns.

* **Critic-Agent:** Optimized to predict $G_{match}^{(t)}$ from the $S_{pre\_match}$ state.
* **Critic-Opponent:** Optimized to predict $G_{play}^{(t)}$ from the $S_{post\_play}$ state.

**The Advantage Updates:**
During PPO backpropagation, the Advantages used to update the Actor heads are calculated using the Temporal Difference (TD) error or Generalized Advantage Estimation (GAE) based on these predictions:

* **Match Advantage:** $A_{match} = G_{match}^{(t)} - V_{agent}(S_{pre\_match})$
* **Play Advantage:** $A_{play} = G_{play}^{(t)} - V_{opponent}(S_{post\_play})$

*(Note on the Play Critic: The Expected Value (EV) weighting across all possible play actions, $\sum \pi(a_i) V_{opponent}(S_{post\_play\_i})$, is maintained as the baseline for calculating the Play Advantage).*
