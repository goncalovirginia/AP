import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.cuda as cuda
from snake_game import SnakeGame
from collections import namedtuple, deque

# Constants

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LEARNING_RATE = 1e-4

# Models

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)   


#ReplayMemory
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

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

# Training

# Get number of actions from gym action space
#n_actions = env.action_space.n (n tinhas acabado dei comment Sofia)
# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)
memory = ReplayMemory(10000)

steps_done = 0

def train():
    snakeGame = SnakeGame(width=16, height=16, food_amount=1, border=1, grass_growth=0.1, max_grass=20)
    board = snakeGame.get_state()
    done = False

    while (not done):
        # feed board into DQN model something something predict next move
        nextMove = 0
        board, reward, done, info = snakeGame.step(nextMove)

        # use variables for something

# Predict

def predict():
    snakeGame = SnakeGame(width=16, height=16, food_amount=1, border=1, grass_growth=0.1, max_grass=20)
    board = snakeGame.get_state()
    done = False

    while (not done):
        # feed board into DQN model something something predict next move
        nextMove = 0
        board, reward, done, info = snakeGame.step(nextMove)

        # use variables for something

