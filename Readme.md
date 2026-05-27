# Reinforcement Learning

Implementation of reinforcement learning algorithms from scratch, organized by method. Each file contains detailed algorithm comments and code structure.

## Environments

- **GridWorld**: Custom 12×4 grid world with forbidden zones and a target, used for discrete control algorithms
- **CartPole-v1**: Inverted pendulum balancing, used for Policy Gradient algorithms
- **ALE/Breakout-v5**: Atari Breakout game, used for deep reinforcement learning algorithms

## Project Structure

```text
├── src/
│   └── grid_world.py          # GridWorld environment
├── examples/
│   └── arguments.py           # GridWorld environment configuration
├── monte_carlo/               # Monte Carlo methods
│   ├── mc_prediction.py       # MC Prediction (policy evaluation)
│   ├── mc_basic.py            # MC basic control
│   ├── mc_exploring_starts.py # MC Exploring Starts
│   └── mc_epsilon_greedy.py   # MC epsilon-greedy control
├── td/                        # Temporal Difference methods
│   ├── td_zero.py             # TD(0) prediction
│   ├── sarsa.py               # Sarsa (on-policy)
│   ├── q_learning.py          # Q-Learning (off-policy)
│   ├── expected_sarsa.py      # Expected Sarsa
│   └── n_step_sarsa.py        # N-step Sarsa
├── function_approximation/    # Function approximation methods
│   ├── td_zero_fa.py          # TD(0) + linear function approximation
│   ├── sarsa_fa.py            # Sarsa + linear function approximation
│   ├── q_learning_fa.py       # Q-Learning + linear function approximation
│   ├── dqn_GridWorld.py       # DQN (GridWorld)
│   ├── dqn_BreakOut.py        # DQN (Breakout)
│   └── utils.py               # Networks, ReplayBuffer, FrameStack, etc.
└── policy_gradient/           # Policy Gradient methods
    ├── reinforce.py           # REINFORCE (MC Policy Gradient)
    ├── actor_critic.py        # Actor-Critic (single-step TD, Breakout)
    ├── a2c.py                 # A2C (Advantage Actor-Critic, CartPole)
    ├── ppo.py                 # PPO (Proximal Policy Optimization, Breakout)
    └── utils.py               # LinearActor/Critic, CNNActor/Critic, etc.
```

## Algorithm Progression

```text
Tabular Methods
  MC Prediction → MC Control (Exploring Starts → epsilon-greedy)
  TD(0) → Sarsa → Q-Learning → Expected Sarsa → N-step Sarsa

Function Approximation
  Linear FA (TD/Sarsa/Q-Learning) → DQN (experience replay + target network)

Policy Gradient
  REINFORCE → Actor-Critic → A2C → PPO
```

## Installation

```bash
pip install gymnasium torch numpy matplotlib opencv-python
pip install gymnasium[atari] ale-py
pip install autorom[accept-rom-license]
AutoROM --accept-license
```

## Usage

```bash
# Monte Carlo
python monte_carlo/mc_prediction.py

# Temporal Difference
python td/q_learning.py

# DQN GridWorld
python function_approximation/dqn_GridWorld.py

# DQN Breakout
python function_approximation/dqn_BreakOut.py

# Policy Gradient
python policy_gradient/reinforce.py
python policy_gradient/a2c.py
python policy_gradient/actor_critic.py
python policy_gradient/ppo.py
```
