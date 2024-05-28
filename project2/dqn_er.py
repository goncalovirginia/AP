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
BATCH_SIZE = 128
GAMMA = 0.99
EPSILON_START = 1.0
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
            Linear(in_features=16 * 8, out_features=64),
            LeakyReLU(),
            Linear(in_features=64, out_features=64),
            LeakyReLU(),
            Linear(in_features=64, out_features=32),
            LeakyReLU(),
            Linear(in_features=32, out_features=n_actions),
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
episode_infos = []

def select_action(state):
    epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * steps_done / EPSILON_DECAY)

    if random.random() > epsilon_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    
    return torch.tensor([[random.choice([0, 1, 2])]], device=device, dtype=torch.long)

def plot_info(episode_infos):
    plt.figure(1)
    plt.title('Episode Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(list(map(lambda episode_info : episode_info.get('score'), episode_infos)))
    plt.show()

def optimize_model(state, action, next_state, reward): 
    state_action_value = policy_net(state).gather(1, action)

    next_state_value = torch.zeros(1, device=device)
    if next_state is not None:
        with torch.no_grad():
            next_state_value = target_net(next_state).max(1).values

    expected_state_action_value = (next_state_value * GAMMA) + reward

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_value, expected_state_action_value.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
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

            optimize_model(state, action, next_state, reward)

            # Soft update of the target network's weights
            # θ′ ← τ θ + (1 −τ )θ′
            target_net_state_dict = target_net.state_dict()
            policy_net_state_dict = policy_net.state_dict()

            for key in policy_net_state_dict:
                target_net_state_dict[key] = policy_net_state_dict[key] * UPDATE_RATE + target_net_state_dict[key] * (1 - UPDATE_RATE)
            target_net.load_state_dict(target_net_state_dict)

            state = next_state

            if done:
                break

        print(f'Episode {episode}: {info}')
        episode_infos.append(info)

# Run stuff

snakeGame = SnakeGame(width=BOARD_WIDTH-2, height=BOARD_HEIGHT-2, food_amount=1, border=1, grass_growth=0.001, max_grass=0.05)

train(snakeGame)

torch.save(policy_net.state_dict(), "policy_net_2.pth")
torch.save(target_net.state_dict(), "target_net_2.pth")

plot_info(episode_infos)