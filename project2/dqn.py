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
import numpy as np
import torchvision.transforms.v2 as transforms
from matplotlib.animation import FuncAnimation
from models import DQN
from replay_memory import Transition, ReplayMemory
from plotting import plot_info, plot_board
from astar import astar, boardToMaze, closestApple, coordsToMove, nextDirection, nextMove

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
EPSILON_END = 0.1
EPSILON_DECAY = 50000
UPDATE_RATE = 0.005
LEARNING_RATE = 0.001
REPLAY_MEMORY_CAPACITY = 10000

# GPU

device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
print(f'Device: {device}')

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

def select_action_astar(state, state_info):
    epsilon_threshold = EPSILON_END + (EPSILON_START - EPSILON_END) * math.exp(-1. * steps_done / EPSILON_DECAY)

    if random.random() > epsilon_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    
    maze = boardToMaze(state, state_info)
    closest_apple = closestApple(state_info[2], state_info[1])
    next_coords = astar(maze, state_info[2], closest_apple)[1]

    if next_coords[0] == -1:
        next_move = random.choice([0, 1, 2])
    else:
        next_move = coordsToMove(state_info[2], next_coords, state_info[4]) + 1

    return torch.tensor([[next_move]], device=device, dtype=torch.long)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

def train(snakeGame):
    for episode in range(NUM_EPISODES):
        state, reward, done, info = snakeGame.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        for step in range(MAX_STEPS):
            #action = select_action(state)
            action = select_action_astar(state, snakeGame.get_state())
            global steps_done
            steps_done += 1
            next_state, reward, done, info = snakeGame.step(action.item() - 1)
            #plot_board(next_state)

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
                break

        print(f'Episode {episode}: {info}')
        episode_infos.append(info)

# Run stuff

snakeGame = SnakeGame(width=BOARD_WIDTH-2, height=BOARD_HEIGHT-2, food_amount=1, border=1, grass_growth=0.001, max_grass=0.01)

train(snakeGame)

torch.save(policy_net.state_dict(), "policy_net_2.pth")
torch.save(target_net.state_dict(), "target_net_2.pth")

plot_info(episode_infos)