import matplotlib.pyplot as plt
import numpy as np

def plot_board(board):
    plt.figure(1)
    plt.ion()
    plt.imshow(board, vmin=0.0, vmax=1.0)
    plt.pause(0.1)

def plot_info(episode_infos):
    plt.figure(2)
    plt.title('Episode Scores')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.plot(list(map(lambda episode_info : episode_info.get('score'), episode_infos)))
    plt.show()
