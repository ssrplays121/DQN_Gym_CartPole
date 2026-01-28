# DQN CartPole with Virtual Constraints

A Deep Q-Network implementation for the CartPole-v1 environment using PyTorch and Gymnasium. This project implements virtual boundary constraints, requiring the agent to balance the pole within a restricted space with heavier penalties for boundary violations.

## **Part of American Centre AI Workshop for RCCIIT**

## Credits

- **Task assigned by:** SNEHASIS BANERJEE, Sr. Scientist, TCS Research; Guest Faculty, IIIT-G; SM: IEEE, CSI, ACM  
- **Based on:** His codebase from [his tutorial page](https://snehasisb.github.io/tutorial/) and his lecture "Nuts and Bolts of AI"
- **Code written and modified by:** Sayantan Saha Roy, CSE Department, RCC Institute of Information Technology

## Features

- Deep Q-Network with experience replay and target networks
- Virtual boundary constraints with enhanced penalty mechanisms
- Efficient numpy-backed replay buffer
- Trained on CartPole-v1 environment

## Setup Instructions

Follow these steps to set up and run the environment:

1. **Create a virtual environment:**

   ``` bash
   python -m venv gymenv
   ```

2. **Activate the virtual environment:**
   - On Linux/macOS:

     ``` bash
     source gymenv/bin/activate
     ```

   - On Windows:

     ``` bash
     source gymenv\Scripts\activate
     ```

3. **Upgrade pip and install dependencies:**

   ``` bash
   pip install --upgrade pip
   pip install torch --index-url https://download.pytorch.org/whl/cpu #Run purely on CPU, no GPU required
   pip install gymnasium[classic-control]
   ```

4. **Run the agent:**

   ``` bash
   python virtual_constraint_dqn.py
   ```

Complete Repository on Github [ssrplays121/DQN_Gym_CartPole](https://github.com/ssrplays121/DQN_Gym_CartPole)