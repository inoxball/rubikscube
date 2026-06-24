# 2x2 Rubik's Cube Solver (Q-learning)

An AI that solves a scrambled 2x2x2 Rubik's Cube (Pocket Cube) on its own, with no human solutions or labelled data. A neural network is trained with **Q-learning** to score every possible move from a cube state, and a **weighted A\*** search uses those scores to find the actual solution.

## How it works

- The cube is a `(6, 2, 2)` array and gets one-hot encoded into a 144-dim vector.
- A residual MLP outputs 18 Q-values (one per move). A move's Q-value is roughly minus the number of moves left after making it.
- Training is pure self-play: scramble cubes, look at the neighbours, and learn from the Bellman target `Q(s,a) = -1 + 0.99 * max Q(s')`, using a target network for stability.
- A **curriculum** starts at 1-move scrambles and works up to 12 as the model improves.
- At solve time, weighted A* (`f = g + W*h`, with `h = -max Q`) searches for the move sequence.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

**Train + see the analysis:** open `analysis.ipynb` and run the cells top to bottom. It trains the model, saves the best weights to `model.pth`, and plots the results.

**Run the web demo:**

```bash
python app.py
```

Then open http://localhost:5000 in a browser to scramble the cube and watch the AI solve it.

## Files

| File | Description |
|------|-------------|
| `cube.py` | 2x2 cube simulator and moves |
| `model.py` | The Q-network |
| `analysis.ipynb` | Training + analysis notebook |
| `app.py` | Flask backend for the web demo |
| `index.html` | Frontend |
| `model.pth` | Trained model weights |
| `report.txt` | Project report |

## Results

The model solves close to 100% of easy and medium scrambles (up to ~7 moves) and around 70-80% of the hardest ones (11-12 moves, near the 2x2 god's number of 11).
