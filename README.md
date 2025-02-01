# Mountain Car Solver with Q-Table and Neural Network

This repository contains Python implementations for solving the classic Mountain Car problem using two approaches:

- Q-Table (Tabular Q-Learning)

- Neural Network (Deep Q-Learning)

## Overview

The Mountain Car problem is a reinforcement learning task where an underpowered car must reach the top of a hill. The challenge is that the car does not have enough power to reach the goal directly and must leverage momentum by oscillating between the hills.

This repository includes:

- A Q-Table implementation for discrete state-action learning

- A Neural Network (Deep Q-Learning) implementation for continuous learning

- Training scripts for both methods

- Evaluation and visualization tools

## Installation

Ensure you have Python installed

## Usage

### Q-Table Approach

Run the Q-learning agent with:
````
python QMountainCar.py
````
### Neural Network Approach

Run the Deep Q-learning agent with:
````
python MountainCar.py
````
Files Structure
````
|── src/
│   |── QMountainCar.py        # Q-Table implementation
│   |── MountainCar.py         # Neural Network implementation
│   |── testNN.py              # File to test the Neural Network after training
|── ML_Project.pdf             # PDF containing the explanation of the work
|── README.md                  # This file
````
## Algorithm Details

### Q-Table Method

Uses tabular Q-learning with discretized states. The agent updates its Q-values using the Bellman equation:

Q(s, a) = Q(s, a) + α [ r + γ max Q(s', a') - Q(s, a) ]

where:

- α is the learning rate

- γ is the discount factor

- r is the reward

- s and s' are the current and next states

- a is the action taken

### Neural Network (Deep Q-Learning)

Uses a neural network to approximate Q-values instead of a table. The model predicts Q-values for all actions given a state, and updates using a loss function based on the Bellman equation.

## Results & Performance

The Q-Table approach works well for small state spaces but struggles with generalization.

The Neural Network approach adapts to continuous states and can generalize better but requires more training time.
