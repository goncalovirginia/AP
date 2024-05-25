import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
from torch.nn import LayerNorm, Linear, ReLU, LeakyReLU, Sequential, Flatten, Conv2d, MaxPool2d
import torch.optim as optim
from torch.optim import AdamW
import torch.nn.functional as F
from snake_game import SnakeGame
from collections import namedtuple, deque
import random
from game_demo import plot_game, plot_state

# Constants

USE_GPU = True
BOARD_WIDTH = 16
BOARD_HEIGHT = 16
N_OBSERVATIONS = BOARD_WIDTH * BOARD_HEIGHT * 3
N_ACTIONS = 3
NUM_EPISODES = 10000
MAX_STEPS = 1000
BATCH_SIZE = 200
GAMMA = 0.99
EPSILON_START = 0.9
EPSILON_END = 0.0
EPSILON_DECAY = 50000
UPDATE_RATE = 0.005
LEARNING_RATE = 0.001
REPLAY_MEMORY_CAPACITY = 10000

# GPU

device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

#Replay Memory

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Models

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()

        self.convolutional_layers = Sequential(
            Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1),
            MaxPool2d(kernel_size=2, stride=2),
            
            Conv2d(in_channels=8, out_channels=8, kernel_size=3, padding=1),
            MaxPool2d(kernel_size=2, stride=2),
        )

        self.fully_connected_layers = Sequential(
            Flatten(),
            Linear(in_features=16 * 8, out_features=32),
            LeakyReLU(),
            Linear(in_features=32, out_features=32),
            LeakyReLU(),
            Linear(in_features=32, out_features=16),
            LeakyReLU(),
            Linear(in_features=16, out_features=n_actions),
        )

    def forward(self, x):
        x = torch.permute(x, (0, 3, 1, 2))
        x = self.convolutional_layers(x)
        x = self.fully_connected_layers(x)
        return x 
    
# Training

policy_net = DQN(N_OBSERVATIONS, N_ACTIONS).to(device)
target_net = DQN(N_OBSERVATIONS, N_ACTIONS).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
memory = ReplayMemory(REPLAY_MEMORY_CAPACITY)

steps_done = 0
episode_durations = []
episode_scores = []

def select_action(state):
    epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * steps_done / EPSILON_DECAY)

    if random.random() > epsilon_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    
    return torch.tensor([[random.choice([0, 1, 2])]], device=device, dtype=torch.long)

def plot_info(show_result=False):
    plt.figure(1)
    plt.title('Episode Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(episode_scores)
    plt.show()

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def train(snakeGame):
    for episode in range(NUM_EPISODES):
        state, reward, done, info = snakeGame.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for step in range(MAX_STEPS):
            action = select_action(state)
            global steps_done
            steps_done += 1
            next_state, reward, done, info = snakeGame.step(action.item() - 1)

            next_state = torch.tensor(next_state, dtype=torch.float32, device=device).unsqueeze(0)
            if done:
                next_state = None

            reward = torch.tensor([reward], device=device)

            memory.push(state, action, next_state, reward)

            optimize_model()

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * UPDATE_RATE + target_net_state_dict[key] * (1 - UPDATE_RATE)
            
            target_net.load_state_dict(target_net_state_dict)

            state = next_state

            if done:
                print(f'Episode {episode}: {info}')
                episode_durations.append(step + 1)
                episode_scores.append(info.get('score'))
                break

# Run stuff

snakeGame = SnakeGame(width=BOARD_WIDTH-2, height=BOARD_HEIGHT-2, food_amount=1, border=1, grass_growth=0.001, max_grass=0.05)

train(snakeGame)

episode_scores_second_half = episode_scores[NUM_EPISODES//2:]
episode_scores_second_half_avg = sum(episode_scores_second_half) / (NUM_EPISODES//2)

torch.save(policy_net.state_dict(), f"policy_net_avgscore{episode_scores_second_half_avg}.pth")
torch.save(target_net.state_dict(), f"target_net_avgscore{episode_scores_second_half_avg}.pth")

plot_info()