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
import os

MAX_MEMORY = 100_0000  # 100_000
BATCH_SIZE = 1000
LR = 0.001
obstacle_radius = 10
# danger_radius = 10


class Agent:

    def __init__(self):
        self.no_of_episodes = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9        # discount rate                            # 0.9 earlier
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.model = Linear_QNet(22, 2048, 8)   # 256

        model_path = '/home/anthrax/Projects/國立中正大學/Project/RL/UAV-Path-Planning/src/model/model.pth'

        if os.path.exists(model_path):
            print("using saved model")
            self.model.load_state_dict(torch.load(model_path))
        else:
            print("Creating new model")

        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def get_state(self, env, idx):
        head = env.head[idx]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)

        point_ul = Point(head.x - 20, head.y - 20)
        point_ur = Point(head.x + 20, head.y - 20)
        point_dl = Point(head.x - 20, head.y + 20)
        point_dr = Point(head.x + 20, head.y + 20)

        dir_l = env.direction[idx] == Direction.LEFT
        dir_r = env.direction[idx] == Direction.RIGHT
        dir_u = env.direction[idx] == Direction.UP
        dir_d = env.direction[idx] == Direction.DOWN

        dir_ul = env.direction[idx] == Direction.UL
        dir_ur = env.direction[idx] == Direction.UR
        dir_dl = env.direction[idx] == Direction.DL
        dir_dr = env.direction[idx] == Direction.DR

        state = [
            # Danger straight
            (dir_r and env.is_collision(point_r, checking_state=True)) or
            (dir_l and env.is_collision(point_l, checking_state=True)) or
            (dir_u and env.is_collision(point_u, checking_state=True)) or
            (dir_d and env.is_collision(point_d, checking_state=True)) or

            (dir_ul and env.is_collision(point_ul, checking_state=True)) or
            (dir_ur and env.is_collision(point_ur, checking_state=True)) or
            (dir_dl and env.is_collision(point_dl, checking_state=True)) or
            (dir_dr and env.is_collision(point_dr, checking_state=True)),

            # Danger right
            (dir_u and env.is_collision(point_r, checking_state=True)) or
            (dir_d and env.is_collision(point_l, checking_state=True)) or
            (dir_l and env.is_collision(point_u, checking_state=True)) or
            (dir_r and env.is_collision(point_d, checking_state=True)) or

            (dir_ul and env.is_collision(point_ur, checking_state=True)) or
            (dir_ur and env.is_collision(point_dr, checking_state=True)) or
            (dir_dr and env.is_collision(point_dl, checking_state=True)) or
            (dir_dl and env.is_collision(point_ul, checking_state=True)),

            # Danger left
            (dir_d and env.is_collision(point_r, checking_state=True)) or
            (dir_u and env.is_collision(point_l, checking_state=True)) or
            (dir_r and env.is_collision(point_u, checking_state=True)) or
            (dir_l and env.is_collision(point_d, checking_state=True)) or

            (dir_ul and env.is_collision(point_dl, checking_state=True)) or
            (dir_dl and env.is_collision(point_dr, checking_state=True)) or
            (dir_dr and env.is_collision(point_ur, checking_state=True)) or
            (dir_ur and env.is_collision(point_ul, checking_state=True)),

            # Danger back
            (dir_d and env.is_collision(point_u, checking_state=True)) or
            (dir_u and env.is_collision(point_d, checking_state=True)) or
            (dir_r and env.is_collision(point_l, checking_state=True)) or
            (dir_l and env.is_collision(point_r, checking_state=True)) or

            (dir_ul and env.is_collision(point_dr, checking_state=True)) or
            (dir_ur and env.is_collision(point_dl, checking_state=True)) or
            (dir_dl and env.is_collision(point_ur, checking_state=True)) or
            (dir_dr and env.is_collision(point_ul, checking_state=True)),

            # Danger diag left
            (dir_ul and env.is_collision(point_l, checking_state=True)) or
            (dir_l and env.is_collision(point_dl, checking_state=True)) or
            (dir_dl and env.is_collision(point_d, checking_state=True)) or
            (dir_d and env.is_collision(point_dr, checking_state=True)) or

            (dir_dr and env.is_collision(point_r, checking_state=True)) or
            (dir_r and env.is_collision(point_ur, checking_state=True)) or
            (dir_ur and env.is_collision(point_u, checking_state=True)) or
            (dir_u and env.is_collision(point_ul, checking_state=True)),

            # danger diag right
            (dir_ul and env.is_collision(point_u, checking_state=True)) or
            (dir_u and env.is_collision(point_ur, checking_state=True)) or
            (dir_ur and env.is_collision(point_r, checking_state=True)) or
            (dir_r and env.is_collision(point_dr, checking_state=True)) or

            (dir_dr and env.is_collision(point_d, checking_state=True)) or
            (dir_d and env.is_collision(point_dl, checking_state=True)) or
            (dir_dl and env.is_collision(point_l, checking_state=True)) or
            (dir_l and env.is_collision(point_ul, checking_state=True)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            dir_ul,
            dir_ur,
            dir_dl,
            dir_dr,

            # Direction of destination
            # destination left
            (env.destination.x < head.x and env.destination.y == head.y),
            # destination right
            (env.destination.x > head.x and env.destination.y == head.y),
            # destination up
            (env.destination.y < head.y and env.destination.x == head.x),
            # destination down
            (env.destination.y > head.y and env.destination.x == head.x),

            (env.destination.y < head.y and env.destination.x <
             head.x),    # destination up left
            (env.destination.y < head.y and env.destination.x >
             head.x),    # destination up right
            (env.destination.y > head.y and env.destination.x <
             head.x),    # destination down left
            (env.destination.y > head.y and env.destination.x >
             head.x),    # destination up right
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # popleft if MAX_MEMORY is reached
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(
                self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, dones = zip(*mini_sample)
        self.trainer.train_step(states, actions, rewards, next_states, dones)
        # for state, action, reward, nexrt_state, done in mini_sample:
        #    self.trainer.train_step(state, action, reward, next_state, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # random moves: tradeoff exploration / exploitation
        self.epsilon = 80 - self.no_of_episodes
        # self.epsilon = 0
        final_move = [0, 0, 0, 0, 0, 0, 0, 0]

        if random.randint(0, 200) < self.epsilon:  # Exploration (0, 200)
            move = random.randint(0, 7)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train(vdo_path, model_path, output_path):

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    # record = 0
    agent = Agent()
    env = Environment()
    model = YOLO(model_path)

    state_old = [None] * env.UAV_Count
    final_move = [None] * env.UAV_Count
    reward = [0] * env.UAV_Count
    state_new = [None] * env.UAV_Count
    done = [False] * env.UAV_Count

    score = [0] * env.UAV_Count
    path_points = [[], [], []]

    frame_count = 0
    no_reached = 0
    total_frames_taken = 0

    while True:

        video_feed = Output(vdo_path, model_path, output_path)

        while video_feed.vdo.isOpened():
            env.obstacle = []
            ret, frame = video_feed.vdo.read()

            if not ret:
                print("Video Over")
                break

            frame_count += 1
            print(f'Frame no is {frame_count}')
            frame_w = video_feed.width//2
            frame_h = video_feed.height//2
            # print(frame_w, frame_h)

            frame = cv2.resize(frame, (frame_w, frame_h))
            results = model.predict(frame, imgsz=(
                frame_h, frame_w), verbose=False)

            box = None

            for result in results:           # len(results == 1)
                box = result.boxes.xywh

            for i in range(env.UAV_Count):
                path_points[i].append(env.head[i])

                if len(path_points[i]) > 1:
                    path_array = np.array(
                        path_points[i], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(frame, [path_array], isClosed=False, color=(
                        0, 255, 0), thickness=2)

            for coor in box:

                yolo_result = coor.tolist()
                env.obstacle.append(
                    Point(int(yolo_result[0]), int(yolo_result[1])))
                cv2.circle(frame, (int(coor[0]), int(
                    coor[1])), obstacle_radius, (0, 0, 255), thickness=3)

                cv2.circle(frame, (int(coor[0]), int(
                    coor[1])), obstacle_radius+10, (255, 0, 0), thickness=2)

            for _h in env.head:
                cv2.circle(frame, (int(_h[0]), int(
                    _h[1])), 7, (255, 255, 0), thickness=-1)

            cv2.circle(frame, (int(env.destination[0]), int(
                env.destination[1])), 15, (255, 0, 0), thickness=-1)

            cv2.imshow('Battlefield', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            # state_old = [None] * env.UAV_Count
            # final_move = [None] * env.UAV_Count
            # reward = [0] * env.UAV_Count
            # state_new = [None] * env.UAV_Count
            # done = [False] * env.UAV_Count

            # score = [0] * env.UAV_Count

            for i in range(env.UAV_Count):
                if not done[i]:
                    state_old[i] = agent.get_state(env, i)

            # get move
            # final_move = [agent.get_action(s) for s in state_old]
            # print("state old", state_old)
            final_move = [None if done[i] else agent.get_action(
                s) for i, s in enumerate(state_old)]

            # print(final_move)
            # perform move and get new state
            _r, _d, _s = env.play_step(final_move)

            for i in range(len(_r)):

                reward[i] = _r[i]
                done[i] = _d[i]
                score[i] = _s[i]

            for i in range(env.UAV_Count):
                # if not done[i]:
                state_new[i] = agent.get_state(env, i)

            # print("state new", state_new)

            # train short memory
            for i in range(env.UAV_Count):
                if final_move[i]:               # Data duplicate
                    agent.train_short_memory(state_old[i], final_move[i], reward[i], state_new[i], done[i])

            # remember
            for i in range(env.UAV_Count):
                if final_move[i]:
                    agent.remember(
                        state_old[i], final_move[i], reward[i], state_new[i], done[i])

            for _ in range(env.UAV_Count):
                if reward[_] == 10:
                    no_reached += 1
                    total_frames_taken += frame_count
                    print(f'{_ + 1} reached after {frame_count} frames.')
                    print(
                        f"Avg frames taken = {total_frames_taken/no_reached}")

            # print(done)
            if all(done):
                frame_count = 0
                # train long memory, plot result
                # print('reset karo')
                env.reset()
                agent.no_of_episodes += 1
                agent.train_long_memory()

                agent.model.save()

                # if score > record:
                #     record = score
                #     agent.model.save()

                # print('Game', agent.no_of_episodes, 'Score', score, 'Record:', record)

                plot_scores.append(sum(score))
                total_score += sum(score)
                mean_score = total_score / agent.no_of_episodes

                # if mean_score > 2:
                #     agent.model.save()

                plot_mean_scores.append(mean_score)
                plot(plot_scores, plot_mean_scores)

                print('Episodes', agent.no_of_episodes,
                      'AvgScore', "{:.4f}".format(mean_score))
                print(score)
                # data_reset()
                state_old = [None] * env.UAV_Count
                final_move = [None] * env.UAV_Count
                reward = [0] * env.UAV_Count
                state_new = [None] * env.UAV_Count
                done = [False] * env.UAV_Count
                path_points = [[], [], []]
                score = [0] * env.UAV_Count

        video_feed.vdo.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':

    vdo_path = '/home/anthrax/Projects/國立中正大學/Project/RL/UAV-Path-Planning/dataset/testvid.mov'
    model_path = r'/home/anthrax/Projects/國立中正大學/Project/UAV-Path-Planning/training/runs/detect/train36-v8l/weights/best.pt'
    output_path = '/home/anthrax/Projects/國立中正大學/Project/RL/UAV-Path-Planning/output/output.mp4'
    train(vdo_path, model_path, output_path)
