import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np
import sys
import math

pygame.init()
font = pygame.font.Font('arial.ttf', 25)
# font = pygame.font.SysFont('arial', 25)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)
GREEN = (170, 120, 120)
OBS_COL = (70, 20, 220)

BLOCK_SIZE = 20
SPEED = 50
# SPEED = sys.maxsize


def calculate_distance(p1, p2):

    distance = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
    return distance


class Environment:

    def __init__(self, w=640, h=480, UAV_Count=2):
        self.w = w
        self.h = h
        # init display
        self.obstacle = []
        self.distane_left = [0] * UAV_Count
        self.UAV_Count = UAV_Count

        # static obstacle

        # self.fixed_obstacles = [(0, self.h - BLOCK_SIZE)]

        # while (0, self.h - BLOCK_SIZE) in self.fixed_obstacles:

        #     self.fixed_obstacles = []
        #     x1 = random.randint(0, w - BLOCK_SIZE)
        #     y1 = random.randint(0, h - BLOCK_SIZE)

        #     x2 = x1 + random.randint(1, BLOCK_SIZE)
        #     y2 = y1

        #     x3 = x1 + random.randint(1, BLOCK_SIZE)
        #     y3 = y1 + random.randint(1, BLOCK_SIZE)

        #     x4 = x1
        #     y4 = y1 + random.randint(1, BLOCK_SIZE)
        #     obst = [Point(x, y) for (x, y) in [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]]

        #     for i in obst:
        #         self.fixed_obstacles.append(i)
    # Return four points forming an irregular object

        for i in range(8):
            obstacle_x = random.randint(0, 640)
            obstacle_y = random.randint(0, 480)
            obstacle_x = obstacle_x // BLOCK_SIZE * BLOCK_SIZE
            obstacle_y = obstacle_y // BLOCK_SIZE * BLOCK_SIZE

            self.obstacle.append(Point(obstacle_x, obstacle_y))

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('UAV')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = [Direction.LEFT, Direction.LEFT]

        # for uav in range(UAV_Count):

        self.head = [Point(self.w - BLOCK_SIZE, 0),
                     Point((self.w // 2 - BLOCK_SIZE), 0)]

        self.score = [0] * self.UAV_Count
        self.destination = None
        self.set_destination()
        self.place_obstacle()
        self.frame_iteration = 0
        self.distane_left = [calculate_distance(
            self.head[0], self.destination), calculate_distance(self.head[1], self.destination)]

    def set_destination(self):
        x = 0
        y = self.h - BLOCK_SIZE
        self.destination = Point(x, y)

    def place_obstacle(self):

        self.obstacle = []
        no_of_obstacles = 4

        while (no_of_obstacles):
            obstacle_x = random.randint(0, 640)
            obstacle_y = random.randint(0, 480)
            obstacle_x = obstacle_x // BLOCK_SIZE * BLOCK_SIZE
            obstacle_y = obstacle_y // BLOCK_SIZE * BLOCK_SIZE

            pt = Point(obstacle_x, obstacle_y)

            if pt != self.destination and pt not in self.head:
                self.obstacle.append(Point(obstacle_x, obstacle_y))
                no_of_obstacles -= 1

    def play_step(self, action):
        self.frame_iteration += 1
        self.place_obstacle()
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. move
        self._move(action)  # update the head

        # 3. check if game over
        reward = [0] * self.UAV_Count

        # Reward according to distance (give reward according to f cost instead)
        for i in range(self.UAV_Count):
            d_left = calculate_distance(self.head[i], self.destination)
            reward[i] += self.distane_left[i] - d_left
            self.distane_left[i] = d_left

        game_over = [False] * self.UAV_Count

        for i in range(self.UAV_Count):
            if self.is_collision(pt=self.head[i]) or self.frame_iteration > 90 * (1 + self.score[i]):
                # print('takkar')
                game_over[i] = True
                reward[i] = -10

        if all(game_over):
            # print('over')
            return reward, game_over, self.score

        # 4. place new destination or just move
        for i in range(self.UAV_Count):
            if self.score[i] != 1 and self.head[i] == self.destination:
                self.score[i] += 1
                game_over[i] = True
                reward[i] = 10
        
        if all(game_over):
            return reward, game_over, self.score

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)
        # 6. return game over and score
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        # hits boundary
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # hits obstacle
        if pt in self.obstacle:  # or pt in self.fixed_obstacles:
            return True

        # if pt in self.fixed_obstacles:
        #     print('fixed obs*******************')
        #     return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for i in range(self.UAV_Count):
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(
                self.head[i].x, self.head[i].y, BLOCK_SIZE, BLOCK_SIZE))

        pygame.draw.rect(self.display, RED, pygame.Rect(
            self.destination.x, self.destination.y, BLOCK_SIZE, BLOCK_SIZE))

        for obs in self.obstacle:
            pygame.draw.rect(self.display, GREEN, pygame.Rect(
                obs.x, obs.y, BLOCK_SIZE, BLOCK_SIZE))

        # for obs in self.fixed_obstacles:
        #     pygame.draw.rect(self.display, OBS_COL, pygame.Rect(
        #         obs.x, obs.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        # [straight, right, left]
        # print(action)
        clock_wise = [Direction.RIGHT, Direction.DOWN,
                      Direction.LEFT, Direction.UP]
        
        # idx = clock_wise.index(self.direction)
        idx = [clock_wise.index(i) for i in self.direction]

        new_dir = [None] * self.UAV_Count

        for i in range(self.UAV_Count):
            if np.array_equal(action[i], [1, 0, 0]):
                new_dir[i] = clock_wise[idx[i]]  # no change
            elif np.array_equal(action[i], [0, 1, 0]):
                next_idx = (idx[i] + 1) % 4
                new_dir[i] = clock_wise[next_idx]  # right turn r -> d -> l -> u
            else:  # [0, 0, 1]
                next_idx = (idx[i] - 1) % 4
                new_dir[i] = clock_wise[next_idx]  # left turn r -> u -> l -> d

            self.direction[i] = new_dir[i]
            # print(new_dir[i], action[i])

            x = self.head[i].x
            y = self.head[i].y
            if self.direction[i] == Direction.RIGHT:
                # print('right')
                x += BLOCK_SIZE
            elif self.direction[i] == Direction.LEFT:
                x -= BLOCK_SIZE
                # print('left')
            elif self.direction[i] == Direction.DOWN:
                y += BLOCK_SIZE
                # print('down')
            elif self.direction[i] == Direction.UP:
                y -= BLOCK_SIZE
                # print('up')

            self.head[i] = Point(x, y)
        # print(self.head)
