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
    UL = 5      # Up Left (Diagonally)
    UR = 6      # Up Right (Diagonally)
    DL = 7      # Down Left (Diagonally)
    DR = 8      # Down Right (Diagonally)


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
SPEED = 100
# SPEED = sys.maxsize


def calculate_distance(p1, p2):

    distance = math.sqrt((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2)
    return distance


class Environment:

    def __init__(self, w=640, h=864, UAV_Count=3):
        self.w = w
        self.h = h
        # init display
        self.obstacle = []
        self.distane_left = [0] * UAV_Count
        self.UAV_Count = UAV_Count
        self.head = []

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

        # for i in range(8):
        #     obstacle_x = random.randint(0, self.w)
        #     obstacle_y = random.randint(0, self.h)
        #     obstacle_x = obstacle_x // BLOCK_SIZE * BLOCK_SIZE
        #     obstacle_y = obstacle_y // BLOCK_SIZE * BLOCK_SIZE

        #     self.obstacle.append(Point(obstacle_x, obstacle_y))

        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('UAV')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # init game state
        self.direction = [Direction.LEFT] * self.UAV_Count

        # for uav in range(UAV_Count):

        self.head = [Point(self.w - BLOCK_SIZE, 10),
                     Point((5 * self.w // 7 - BLOCK_SIZE), 10), Point((3 * BLOCK_SIZE), 10)]

        # self.head = [Point(10, 40)]

        self.score = [0] * self.UAV_Count
        self.destination = None
        self.set_destination()
        # self.place_obstacle()
        self.frame_iteration = 0
        self.distane_left = [calculate_distance(
            self.head[i], self.destination) for i in range(self.UAV_Count)]

    def set_destination(self):
        # x = 0
        # y = self.h - BLOCK_SIZE
        # self.destination = Point(530 , 780)
        x = random.randint(0, self.w)
        y = random.randint(0, self.h)
        self.destination = Point(x, y)
        # self.destination = Point(self.w - 150, 150)
        # self.destination = Point(self.w - 150, self.h - 150)

    def place_obstacle(self):   # now works as clear obstacles rathar rename later

        self.obstacle = []
        # no_of_obstacles = 4

        # while (no_of_obstacles):
        #     obstacle_x = random.randint(0, self.w)
        #     obstacle_y = random.randint(0, self.h)
        #     obstacle_x = obstacle_x // BLOCK_SIZE * BLOCK_SIZE
        #     obstacle_y = obstacle_y // BLOCK_SIZE * BLOCK_SIZE

        #     pt = Point(obstacle_x, obstacle_y)

        #     if pt != self.destination and pt not in self.head:
        #         self.obstacle.append(Point(obstacle_x, obstacle_y))
        #         no_of_obstacles -= 1

    def play_step(self, action):
        self.frame_iteration += 1
        # self.place_obstacle()
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
            # /self.distane_left[i] * 10  # Normal acc to distance covered
            reward[i] += (self.distane_left[i] - d_left)
            self.distane_left[i] = d_left

            distances = []
            for obstacle in self.obstacle:
                d = calculate_distance(self.head[i], obstacle)
                distances.append((obstacle, d))

            # # Sort the distances in ascending order
            distances.sort(key=lambda x: x[1])

            # # Extract the nearest three obstacles
            nearest_obstacles = distances[:3]

            total_distance_of_obstacles = 0
            for obstacle, dist in nearest_obstacles:
                # print(f"Obstacle at ({obstacle.x}, {obstacle.y}) - Distance: {dist}")
                total_distance_of_obstacles += dist
            print(total_distance_of_obstacles)
            reward[i] += (5 * total_distance_of_obstacles/150)    # Change 5 and 150 accordingly (150 = 50 * 3)

            # reward[i] -= (150 / (total_distance_of_obstacles+1))    # Change 5 and 150 accordingly (150 = 50 * 3)
            # reward[i] = max(reward[i], 0)

        # game_over = [False] * self.UAV_Count

        game_over = [True if action[i] ==
                     None else False for i in range(self.UAV_Count)]

        for i in range(self.UAV_Count):
            if not game_over[i]:
                # was 90
                if self.is_collision(pt=self.head[i], game_over=game_over) or self.frame_iteration > 110 * (1 + self.score[i]):
                    # print('takkar')
                    game_over[i] = True
                    reward[i] = -10

        if all(game_over):
            # print('over', game_over)
            return reward, game_over, self.score

        # 4. just move
        for i in range(self.UAV_Count):
            if not game_over[i]:
                # if self.score[i] != 1 and self.head[i] == self.destination:
                if self.score[i] != 1 and calculate_distance(self.head[i], self.destination) < 30:
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

    def is_collision(self, pt=None, game_over=None, checking_state=False):
        # if pt is None:
        #     pt = self.head
        # hits boundary
        collision_distance = 30
        collision_distance_self = 10
        if checking_state:
            collision_distance = 50
            collision_distance_self = 20

        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True

        # hits obstacle
        # if pt in self.obstacle:  # or pt in self.fixed_obstacles:
        #     return True

        for obst in self.obstacle:
            if calculate_distance(pt, obst) <= collision_distance:    # play with values
                return True

        # collision with other UAVs
        _c = 0
        for h in self.head:   # TODO Wrong check distance rathar
            idx = self.head.index(h)
            if pt == h and game_over and not game_over[idx]:
                if calculate_distance(pt, h) <= collision_distance_self:
                    _c += 1

        if _c == 2:
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
        # [straight, right, left, diag_left, diag_right, back, back_right, back_left]
        # print('before', self.head)
        clock_wise = [Direction.RIGHT, Direction.DR, Direction.DOWN,
                      Direction.DL, Direction.LEFT, Direction.UL, Direction.UP, Direction.UR]

        # idx = clock_wise.index(self.direction)
        idx = [clock_wise.index(i) for i in self.direction]

        new_dir = [None] * self.UAV_Count

        for i in range(self.UAV_Count):

            if action[i] == None:
                continue

            if np.array_equal(action[i], [1, 0, 0, 0, 0, 0, 0, 0]):
                new_dir[i] = clock_wise[idx[i]]  # no change

            elif np.array_equal(action[i], [0, 1, 0, 0, 0, 0, 0, 0]):
                next_idx = (idx[i] + 2) % 8
                # right turn r -> d -> l -> u
                new_dir[i] = clock_wise[next_idx]

            elif np.array_equal(action[i], [0, 0, 1, 0, 0, 0, 0, 0]):

                next_idx = (idx[i] - 2) % 8
                new_dir[i] = clock_wise[next_idx]  # left turn r -> u -> l -> d

            elif np.array_equal(action[i], [0, 0, 0, 1, 0, 0, 0, 0]):
                next_idx = (idx[i] - 1) % 8
                new_dir[i] = clock_wise[next_idx]

            elif np.array_equal(action[i], [0, 0, 0, 0, 1, 0, 0, 0]):
                next_idx = (idx[i] + 1) % 8
                new_dir[i] = clock_wise[next_idx]

            elif np.array_equal(action[i], [0, 0, 0, 0, 0, 1, 0, 0]):
                next_idx = (idx[i] + 4) % 8
                new_dir[i] = clock_wise[next_idx]

            elif np.array_equal(action[i], [0, 0, 0, 0, 0, 0, 1, 0]):
                next_idx = (idx[i] + 3) % 8
                new_dir[i] = clock_wise[next_idx]

            elif np.array_equal(action[i], [0, 0, 0, 0, 0, 0, 0, 1]):
                next_idx = (idx[i] + 5) % 8
                new_dir[i] = clock_wise[next_idx]

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

            elif self.direction[i] == Direction.UL:
                y -= BLOCK_SIZE
                x -= BLOCK_SIZE

            elif self.direction[i] == Direction.UR:
                y -= BLOCK_SIZE
                x += BLOCK_SIZE

            elif self.direction[i] == Direction.DL:
                y += BLOCK_SIZE
                x -= BLOCK_SIZE

            elif self.direction[i] == Direction.DR:
                y += BLOCK_SIZE
                x += BLOCK_SIZE

            self.head[i] = Point(x, y)
        # print('after', self.head)
        # print(self.head)
