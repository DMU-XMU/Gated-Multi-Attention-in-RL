# Gated Multi-Attention Representation in Reinforcement learning

This implementation contains:

1. Deep Q-network(DQN)
    - vanilla DQN model
2. RS-DQN
    - DQN model with region-sensitive(RS) module
3. Local-DQN
    - DQN model with local attention(a glimpse network) module
4. ALSTM
    - Attention combined with LSTM based on DQN
5. GMAQN
    - Our work
    
## Dependencies

* run the command `pip3 install -r requirements.txt` and install all the required packages.

## Training

To train on a local machine or in a local container, run the following command:
To train GAMQN model for Seaquest:

    $ python train.py --env Seaquest-v4 --model GMAQN
    
To train ALSTM model for Seaquest:

    $ python train.py --env Seaquest-v4 --model ALSTM

# Grad-CAM visualization videos

![image](https://github.com/DMU-XMU/Gated-Multi-Attention-in-RL/blob/main/Visualize/seaquest_agent1.gif)
![image](https://github.com/DMU-XMU/Gated-Multi-Attention-in-RL/blob/main/Visualize/seaquest_agent2.gif)
![image](https://github.com/DMU-XMU/Gated-Multi-Attention-in-RL/blob/main/Visualize/seaquest_agent3.gif)


Take the Seaquest environment in Atari 2600 games as an example.Our agent receives visual input as a stream of 210x160px RGB images (top).Grad-CAM can mark the
regions of evidence for the current action in each frame via heat. The heat maps can clearly show he current ehavior
and ffensive policy of the agent.

![image](https://github.com/DMU-XMU/Gated-Multi-Attention-in-RL/blob/main/Visualize/heat_maps.png)

In the heat maps, we also show how GMAQN can be trained to supplement oxygen after the agent is aware that oxygen is insufficient. In more detail, in the first picture, the submarine is destroying the enemy, while in the second, third, and fourth pictures,  the agent observed oxygen is depleting. The fifth and sixth pictures show that the submarine floats to the surface to supplement oxygen. In the seventh picture, the submarine starts to destroy the enemy after replenishing oxygen.




















