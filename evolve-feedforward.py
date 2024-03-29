"""
Single-pole balancing experiment using a feed-forward neural network.
"""

from __future__ import print_function

import os
import pickle


import game

import neat
import visualize

import pygame

GAME_TICKS = 1000

X_LIMIT = 800
Y_LIMIT = 800

NO_PYGAME = False

BACKGROUND_COLOR = (125, 125, 125)


pygame.init()
screen = pygame.display.set_mode((X_LIMIT, Y_LIMIT))

pygame.font.init()
myfont = pygame.font.SysFont('Comic Sans MS', 12)

slow_motion = False
num_sheeps = 1


def eval_genomes(genomes, config):
    
    global slow_motion
    global num_sheeps

    nets = []

    num_wolves = 0
    for genome_id, genome in genomes:
        nets.append(neat.nn.FeedForwardNetwork.create(genome, config))
        num_wolves += 1
    
    eval_game = game.Game(num_wolves, screen, num_sheeps)
    
    i = 0
    for genome_id, genome in genomes:
        eval_game.set_genome_id(i, genome_id)
        i += 1

    while eval_game.ticks < GAME_TICKS:
        i = 0
        screen.fill(BACKGROUND_COLOR)

        for net in nets:
            inputs = eval_game.get_scaled_state(i)
            action = net.activate(inputs)
            eval_game.apply_action(action, i)
            i += 1

        eval_game.step()

        i = 0
        for genome_id, genome in genomes:
            genome.fitness = eval_game.get_fitness(i)  
            i += 1

        if NO_PYGAME == False: 
            
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_DOWN:
                        if num_sheeps > 1:
                            num_sheeps -= 1
                    else:
                        if event.key == pygame.K_UP:
                            num_sheeps +=1
                        else:
                            if slow_motion == True:
                                slow_motion = False
                            else:
                                slow_motion = True
                if event.type == pygame.QUIT:
                    return

            if slow_motion == True:
                pygame.time.wait(100)
           
            for wolf in eval_game.wolves:
                wolf.draw(screen)
            for sheep in eval_game.sheeps:
                sheep.draw(screen)

            i = 0

            # for wolf in eval_game.wolves:
            #     text = myfont.render(str(int(wolf.energy)), False, (0, 0, 0))
            #     screen.blit(text,(0,i))
            #     i += 11
            
            # i += 11
            text = myfont.render(str(eval_game.ticks), False, (0, 0, 0))
            screen.blit(text,(0,i))

            pygame.display.update()

def run():

    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    # config_path = os.path.join(local_dir, 'config-feedforward')
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    checkpointer = neat.Checkpointer(5)
    # pop = checkpointer.restore_checkpoint("neat-checkpoint-394-recurrent")
    pop = checkpointer.restore_checkpoint("neat-checkpoint-999-feedforward")

    # pop = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.Checkpointer(5))

    winner = pop.run(eval_genomes, 200)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    pygame.quit()

    visualize.plot_stats(stats, ylog=True, view=True, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=True, filename="feedforward-speciation.svg")

    node_names = {-1: 'x', -2: 'dx', -3: 'theta', -4: 'dtheta', 0: 'control'}
    visualize.draw_net(config, winner, True, node_names=node_names)

    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward.gv")
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled.gv", show_disabled=False)
    visualize.draw_net(config, winner, view=True, node_names=node_names,
                       filename="winner-feedforward-enabled-pruned.gv", show_disabled=False, prune_unused=True)


if __name__ == '__main__':
    run()
