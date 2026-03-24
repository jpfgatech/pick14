Poker Agent Action Network: Behavioral Cloning Architecture
1. State Representation (Input Encoding)

The environment consists of up to 7 hand combinations and a dynamically sized pool of public cards, plus a single "Pass" option.

Card Combinations (9-Dimensional Vector):
Each valid hand combination and each public card is initially represented by a sparse 9-element vector:

Sum of card numbers

Sum of card points

Card 1 number

Card 1 points

Card 2 number

Card 2 points

Card 3 number

Card 3 points

Total number of cards in the combination
(Note: For combinations with fewer than 3 cards, unused slots are zero-padded. Public cards are treated as single-card combinations).

The "Pass" Action:
The Pass action bypasses the 9-dimensional encoding entirely. It is represented by a standalone, independently trainable 32-dimensional parameter vector.

2. Network Topology (Pointer/Attention Head)

The network abandons a traditional flat output layer in favor of a set-based, size-invariant Attention mechanism.

Embedding Layer: A single linear transformation (followed by a ReLU activation) that projects the raw 9-dimensional card vectors into a richer 32-dimensional latent space.

Query and Key Projections: * The 7 embedded hand combinations are multiplied by a learned Query matrix.

The pool of embedded public cards (plus the appended 32-dimensional Pass vector) are multiplied by a learned Key matrix.

Scoring: The network computes the dot product between the Queries and Keys, resulting in a raw score matrix representing the strategic value of every possible pairing.

3. Rule Injection (Action Masking)

To ensure zero sample-inefficiency on illegal moves, deterministic game rules are injected directly into the network's forward pass before action sampling.

The 14-Rule: A pairing is only mathematically valid if:
Hand Combination Number + Public Card Number == 14.

The Exception: The "Pass" action pairing is unconditionally valid.

The Mask: A binary mask evaluates all possible Query/Key pairs. Any invalid pair has its raw network score replaced with negative infinity (−∞).

4. Output and Training (Imitation Phase)

Softmax: The masked score matrix is flattened and passed through a Softmax function. Because invalid actions were set to −∞, they receive a mathematical probability of exactly 0%.

Loss Function (Cross-Entropy): During this initial training phase, the network's output probability distribution is evaluated against the single, deterministic integer choice provided by the algorithmic Greedy Teacher. The network updates its embedding and attention weights to mimic this greedy strategy.
