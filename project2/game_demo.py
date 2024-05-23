#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 27 07:35:58 2021
Updated on Fri May 12 07:35:00 2023

"""

import matplotlib.pyplot as plt
from snake_game import SnakeGame

def plot_game(transitions):
    plt.ion()
    plt.figure(num=1, figsize=(5,5))

    for transition in transitions:
        plt.imshow(transition.state.cpu().numpy()[0])
        plt.axis('off')
        plt.pause(0.1)

def plot_state(state):
    plt.ion()
    plt.figure(num=1, figsize=(5, 5))

    plt.imshow(state.cpu().numpy()[0])
    plt.axis('off')
    plt.pause(0.05)
    plt.gcf()

def plot_board(file_name,board,text=None):
    plt.figure(figsize=(10,10))
    plt.imshow(board)
    plt.axis('off')
    if text is not None:
        plt.gca().text(3, 3, text, fontsize=45,color = 'yellow')
    plt.savefig(file_name,bbox_inches='tight')
    plt.close()

def snake_demo(actions):
    game = SnakeGame(30,30,border=1, max_grass=10)
    board,reward,done,info = game.reset()    
    action_name = {-1:'Turn left',0:'Straight ahead',1:'Turn right'}    
    plot_board('project2/images/0.png',board,'Start')
    for frame,action in enumerate(actions):
        board,reward,done,info = game.step(action)
        plot_board(f'project2/images/{frame+1}.png',board,action_name[action])
        
snake_demo([0,1,0,-1,0])
    
