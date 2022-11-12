#!python

import pygame
import math
import sys
import random
import numpy as np

BIG_NUMBER = 100000000.0

SPEED_DELTA_ANGLE_THRESHOLD = 2*np.pi/16
MAGNITUDE_THRESHOLD = 8

WOLF_INITIAL_ENERGY = 1000

BACKGROUND_COLOR = (125, 125, 125)
WOLVES_COLOR = (0, 0, 0)
SHEEPS_COLOR = (255, 255, 255)

X_LIMIT = 800
Y_LIMIT = 800

WOLF_SPEED_LIMIT = 7.0
WOLF_ACCEL_LIMIT = 2.0
WOLF_MAGNITUDE_SCALING = 0.5
WOLF_ANGLE_SCALING = 0.5

SHEEP_SPEED_LIMIT = 7.0
SHEEP_ACCEL_LIMIT = 2.0
SHEEP_MAGNITUDE_SCALING = 0.5
SHEEP_ANGLE_SCALING = 0.5

WOLVES_RADIUS = 3

MAX_SHEEPS = 10
SHEEPS_RADIUS = 3

# ------------ Game ------------

class Game(object):
    def __init__(self, num_wolves, screen, num_sheeps):
        self.ticks = 0
        self.prev_best = 0
        self.screen = screen
        
        self.num_wolves = num_wolves
        self.wolves = []
        for count in range(0,self.num_wolves):
            x = Wolf()
            self.wolves.append(x)

        self.sheeps = []
        for count in range(0,num_sheeps):
            x = Sheep()
            self.sheeps.append(x)

    def set_genome_id(self, i, genome_id):
        self.wolves[i].genome_id = genome_id

    def get_fitness(self, i):
        return self.wolves[i].energy

    def get_scaled_state(self, i):
        min_magnitude = BIG_NUMBER
        min_delta_x = BIG_NUMBER
        min_delta_y = BIG_NUMBER
        # self.wolves[i].energy -= 1
        delta_angle = 0
        min_sheep = Sheep()
        for sheep in self.sheeps:
            delta_x = sheep.x-self.wolves[i].x
            if delta_x > X_LIMIT/2: # per la distanza sul toroide
                delta_x -= X_LIMIT
            delta_y = sheep.y-self.wolves[i].y
            if delta_y > Y_LIMIT/2: # per la distanza sul toroide
                delta_y -= Y_LIMIT
            magnitude = np.linalg.norm([delta_x, delta_y])
            if magnitude < min_magnitude:
                min_delta_x = delta_x
                min_delta_y = delta_y
                min_magnitude = magnitude
                direction_vector = [min_delta_x, min_delta_y]
                wolf_vector = [self.wolves[i].x_speed, self.wolves[i].y_speed]
                delta_angle = np.arccos(np.dot(direction_vector,wolf_vector)/(np.linalg.norm(direction_vector)*np.linalg.norm(wolf_vector)))
                min_sheep = sheep
        if min_magnitude < MAGNITUDE_THRESHOLD and (delta_angle < SPEED_DELTA_ANGLE_THRESHOLD or delta_angle > np.pi*2-SPEED_DELTA_ANGLE_THRESHOLD):
            self.wolves[i].energy += 100
            if self.wolves[i].energy > self.wolves[self.prev_best].energy:
                self.wolves[self.prev_best].best = False
                self.wolves[i].best = True
                self.prev_best = i
            min_sheep.reset()
        if self.wolves[i].best == True:
            pygame.draw.line(self.screen, (255, 0, 0), (int(self.wolves[i].x),int(self.wolves[i].y)),(int(self.wolves[i].x+min_delta_x),int(self.wolves[i].y+min_delta_y)))

        self.wolves[i].energy -= min_magnitude/X_LIMIT
        return [min_delta_x/X_LIMIT*2, min_delta_y/Y_LIMIT*2, delta_angle/2/np.pi]

    # action [0,1]
    def apply_action(self, action, i):
        self.wolves[i].update(action[0], action[1])
        

    def step(self):
        for sheep in self.sheeps:
            magnitiude_accel_delta = random.random()
            angle_accel_delta = random.random()
            sheep.update(magnitiude_accel_delta, angle_accel_delta)
        self.ticks += 1

# ------------ Sheep ------------

class Sheep:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x = random.randrange(0, X_LIMIT)
        self.y = random.randrange(0, Y_LIMIT)
        self.x_speed = random.random()*SHEEP_SPEED_LIMIT
        self.y_speed = random.random()*SHEEP_SPEED_LIMIT
        self.magnitude_accel = 0.0
        self.angle_accel = 0.0

    # valori fra magnitude_accel_delta [0,1] angle_accel_delta [0,1]
    def update(self, magnitude_accel_delta, angle_accel_delta):       
        magnitude_accel_delta = (magnitude_accel_delta*2-1)*SHEEP_MAGNITUDE_SCALING
        angle_accel_delta = (angle_accel_delta*2-1)*SHEEP_ANGLE_SCALING

        self.magnitude_accel += magnitude_accel_delta
        if self.magnitude_accel > SHEEP_ACCEL_LIMIT:
            self.magnitude_accel = SHEEP_ACCEL_LIMIT
        if self.magnitude_accel < 0.0:
            self.magnitude_accel = 0.0

        self.angle_accel += angle_accel_delta
        if self.angle_accel > np.pi:
            self.angle_accel -= 2*np.pi
        if self.angle_accel < -np.pi:
            self.angle_accel += 2*np.pi

        x_accel = self.magnitude_accel * np.cos(self.angle_accel)
        y_accel = self.magnitude_accel * np.sin(self.angle_accel)

        self.x_speed += x_accel
        self.y_speed += y_accel

        magnitude_speed = np.sqrt(self.x_speed**2 + self.y_speed**2)
        if magnitude_speed > SHEEP_SPEED_LIMIT:
            self.x_speed *= SHEEP_SPEED_LIMIT / magnitude_speed
            self.y_speed *= SHEEP_SPEED_LIMIT / magnitude_speed

        self.x += self.x_speed
        if self.x > X_LIMIT:
            self.x = 0
        if self.x < 0:
            self.x = X_LIMIT-1
        
        self.y += self.y_speed
        if self.y > Y_LIMIT:
            self.y = 0
        if self.y < 0:
            self.y = Y_LIMIT-1

    def draw(self, screen):
        pygame.draw.circle(screen, SHEEPS_COLOR, (int(self.x),int(self.y)), SHEEPS_RADIUS)
        pygame.draw.line(screen, SHEEPS_COLOR, (int(self.x),int(self.y)),(int(self.x+self.x_speed),int(self.y+self.y_speed)))


# ------------ Wolf ------------

class Wolf:
    def __init__(self):
        self.x = random.randrange(0, X_LIMIT)
        self.y = random.randrange(0, Y_LIMIT)
        self.x_speed = random.random()*WOLF_SPEED_LIMIT
        self.y_speed = random.random()*WOLF_SPEED_LIMIT
        self.magnitude_accel = 0.0
        self.angle_accel = 0.0
        self.x_direction = self.x_speed
        self.y_direction = self.y_speed
        self.energy = WOLF_INITIAL_ENERGY
        self.genome_id = 0
        self.best = False


    # valori fra magnitude_accel_delta [0,1] angle_accel_delta [0,1]
    def update(self, magnitude_accel_delta, angle_accel_delta):
        magnitude_accel_delta = (magnitude_accel_delta*2-1)*WOLF_MAGNITUDE_SCALING
        angle_accel_delta = (angle_accel_delta*2-1)*WOLF_ANGLE_SCALING

        self.magnitude_accel += magnitude_accel_delta
        if self.magnitude_accel > WOLF_ACCEL_LIMIT:
            self.magnitude_accel = WOLF_ACCEL_LIMIT
        if self.magnitude_accel < 0.0:
            self.magnitude_accel = 0.0

        self.angle_accel += angle_accel_delta
        if self.angle_accel > np.pi:
            self.angle_accel -= 2*np.pi
        if self.angle_accel < -np.pi:
            self.angle_accel += 2*np.pi

        x_accel = self.magnitude_accel * np.cos(self.angle_accel)
        y_accel = self.magnitude_accel * np.sin(self.angle_accel)

        self.x_speed += x_accel
        self.y_speed += y_accel

        magnitude_speed = np.sqrt(self.x_speed**2 + self.y_speed**2)
        if magnitude_speed > WOLF_SPEED_LIMIT:
            self.x_speed *= WOLF_SPEED_LIMIT / magnitude_speed
            self.y_speed *= WOLF_SPEED_LIMIT / magnitude_speed

        if self.x_speed != 0.0 or self.y_speed != 0.0:
            self.x_direction = self.x_speed
            self.y_direction = self.y_speed 

        self.x += self.x_speed
        if self.x > X_LIMIT:
            self.x = 0
        if self.x < 0:
            self.x = X_LIMIT-1
        
        self.y += self.y_speed
        if self.y > Y_LIMIT:
            self.y = 0
        if self.y < 0:
            self.y = Y_LIMIT-1

    def draw(self, screen):
        if self.best == True:
            color = (255,255,0)
        else:
            color = WOLVES_COLOR
        pygame.draw.circle(screen, color, (int(self.x),int(self.y)), WOLVES_RADIUS)
        pygame.draw.line(screen, color, (int(self.x),int(self.y)),(int(self.x+self.x_speed),int(self.y+self.y_speed)))


def main():
    
    pygame.init()
    screen = pygame.display.set_mode((X_LIMIT, Y_LIMIT))

    pygame.font.init()
    myfont = pygame.font.SysFont('Comic Sans MS', 20)

    local_game = Game(30, screen, MAX_SHEEPS)
    
    try:
        while 1:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    pygame.time.wait(10000)
                if event.type == pygame.QUIT:
                    return
            
              

            for i in range(0, local_game.num_wolves):
                local_game.apply_action([random.random(), random.random()], i)
                
            local_game.step()

            screen.fill(BACKGROUND_COLOR)
            
            for wolf in local_game.wolves:
                wolf.draw(screen)
            for sheep in local_game.sheeps:
                sheep.draw(screen)

            for i in range(0, local_game.num_wolves):
                # print(local_game.get_scaled_state(i))
                state = local_game.get_scaled_state(i)
                # text1 = myfont.render(str(state[0]), False, (0, 0, 0))
                # text2 = myfont.render(str(state[1]), False, (0, 0, 0))
                # text3 = myfont.render(str(state[2]), False, (0, 0, 0))
                # screen.blit(text1,(0,0))
                # screen.blit(text2,(0,20))
                # screen.blit(text3,(0,40))
            i = 0

            for wolf in local_game.wolves:
                text = myfont.render(str(wolf.energy), False, (0, 0, 0))
                screen.blit(text,(0,i))
                i += 15
            
            text = myfont.render(str(local_game.ticks), False, (0, 0, 0))
            screen.blit(text,(0,i))
            
            pygame.display.update()
    finally:
        pygame.quit()

if __name__ == '__main__':
    main()
