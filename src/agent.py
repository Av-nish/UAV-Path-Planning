import torch
import random
import numpy as np
from collections import deque
from environment import Environment, Direction, Point
from model import Linear_QNet, QTrainer
from helper import plot
import cv2
from ultralytics import YOLO
from output import Output

MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:

    def __init__(self):
        self.no_of_episodes = 0
        self.epsilon = 0 # randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)


    def get_state(self, env, idx):
        head = env.head[idx]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = env.direction[idx] == Direction.LEFT
        dir_r = env.direction[idx] == Direction.RIGHT
        dir_u = env.direction[idx] == Direction.UP
        dir_d = env.direction[idx] == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and env.is_collision(point_r)) or 
            (dir_l and env.is_collision(point_l)) or 
            (dir_u and env.is_collision(point_u)) or 
            (dir_d and env.is_collision(point_d)),

            # Danger right
            (dir_u and env.is_collision(point_r)) or 
            (dir_d and env.is_collision(point_l)) or 
            (dir_l and env.is_collision(point_u)) or 
            (dir_r and env.is_collision(point_d)),

            # Danger left
            (dir_d and env.is_collision(point_r)) or 
            (dir_u and env.is_collision(point_l)) or 
            (dir_r and env.is_collision(point_u)) or 
            (dir_l and env.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Direction of destination
            env.destination.x < head.x,  # destination left
            env.destination.x > head.x,  # destination right
            env.destination.y < head.y,  # destination up
            env.destination.y > head.y  # destination down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done)) # popleft if MAX_MEMORY is reached

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE) # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        #for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.no_of_episodes
        final_move = [0,0,0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(vdo_path, model_path, output_path):

    output = Output(vdo_path, model_path, output_path)
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    # record = 0
    agent = Agent()
    env = Environment()
    while True:

        state_old = [None] * env.UAV_Count
        final_move = [None] * env.UAV_Count
        reward = [0] * env.UAV_Count
        done = [False] * env.UAV_Count
        # print("done ka size", len(done))
        score = [0] * env.UAV_Count
        state_new = [None] * env.UAV_Count

        for i in range(env.UAV_Count):
            state_old[i] = agent.get_state(env, i)


        # get move
        final_move = [agent.get_action(s) for s in state_old]
        # print(final_move)

        # perform move and get new state
        _r, _d, _s = env.play_step(final_move)
        
        for i in range(len(_r)):

            reward[i] = _r[i]
            done[i] = _d[i]
            score[i] = _s[i]

        for i in range(env.UAV_Count):
            state_new[i] = agent.get_state(env, i)


        # train short memory
        for i in range(env.UAV_Count):
            agent.train_short_memory(state_old[i], final_move[i], reward[i], state_new[i], done[i])

        # remember
        for i in range(env.UAV_Count):
            agent.remember(state_old[i], final_move[i], reward[i], state_new[i], done[i])

        # print(done)
        if all(done):
            # train long memory, plot result
            # print('reset karo')
            env.reset()
            agent.no_of_episodes += 1
            agent.train_long_memory()

            # if score > record:
            #     record = score
            #     agent.model.save()

            # print('Game', agent.no_of_episodes, 'Score', score, 'Record:', record)

            plot_scores.append(sum(score))
            total_score += sum(score)
            mean_score = total_score / agent.no_of_episodes
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)

            print('Episodes', agent.no_of_episodes, 'AvgScore', "{:.4f}".format(mean_score))
            print(score)


if __name__ == '__main__':

    vdo_path = '/home/anthrax/Projects/國立中正大學/Project/UAV-Path-Planning/data/video/video_for_test/testvid7.mov'
    model_path = r'/home/anthrax/Projects/國立中正大學/Project/UAV-Path-Planning/training/runs/detect/train36-v8l/weights/best.pt'
    output_path = '/home/anthrax/Projects/國立中正大學/Project/RL/UAV-Path-Planning/output/output.mp4'
    train(vdo_path, model_path, output_path)
