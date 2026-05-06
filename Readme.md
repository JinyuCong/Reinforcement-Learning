# Reinforcement Learning Practice

Personal practice code based on the textbook and course
**[Mathematical Foundation of Reinforcement Learning](https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning)**
by Shiyu Zhao.

## Structure

```
Code for grid world/python_version/
├── src/
│   └── grid_world.py          # Grid world environment
├── examples/
│   ├── arguments.py           # Environment configuration
│   └── example_grid_world.py  # Basic usage example
├── monte_carlo/
│   ├── utils.py               # Shared utilities (state_to_index, generate_episode)
│   ├── mc_prediction.py       # Every-Visit MC Prediction
│   ├── mc_basic.py            # MC Basic
│   ├── mc_exploring_starts.py # MC Exploring Starts
│   └── mc_epsilon_greedy.py   # MC ε-Greedy (On-Policy Control)
└── td/
    ├── utils.py               # Shared utilities
    ├── td_zero.py             # TD(0) — state value estimation
    ├── sarsa.py               # Sarsa — on-policy TD control
    └── q_learning.py          # Q-Learning — off-policy TD control
```

## Environment

5×5 Grid World with:
- Start: `(2, 2)`, Target: `(4, 4)`
- Forbidden states: `(2,1)`, `(3,3)`, `(1,3)`
- Actions: up / down / left / right / stay
- Rewards: target `+10`, forbidden/boundary `-5`, step `-1`

## Usage

Run any algorithm directly:

```bash
cd "Code for grid world/python_version"
python monte_carlo/mc_basic.py
python td/sarsa.py
```

## Reference

- Textbook repo: https://github.com/MathFoundationRL/Book-Mathematical-Foundation-of-Reinforcement-Learning
