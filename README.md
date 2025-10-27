# Simple Reinforcement Learning Configuration
This small Python project is something I built to practice how reinforcement learning can be used for simple dynamic configuration problems.  
It’s not meant to be a full research model, just a personal prototype to test the main loop of RL: state, action, reward, update.

The idea comes from thinking about how an agent could learn to choose algorithm settings depending on changing conditions.  

## What it does?
The script creates a very small artificial environment.  
The agent gets two random numbers as the state, picks one of three actions, and receives a reward based on how good that choice was.  
It then updates its policy parameters to improve future actions.

The results are noisy and unstable, which is expected in such a simple setup, but it still shows learning behavior after some episodes.

## How to run:
Just open a terminal and run.

## Next steps
I plan to connect this toy environment to a real solver or optimization problem, so the reward will come from actual algorithm performance instead of random values.
