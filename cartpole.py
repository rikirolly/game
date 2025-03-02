import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import random

# Configurazione del device: usa MPS se disponibile (MacOS con chip Apple Silicon), altrimenti CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Utilizzo del device: {device}")

# Definizione dell'environment CartPole usando pygame
class CartPoleEnv:
    def __init__(self, render_mode=False):
        # Parametri fisici e di simulazione
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masscart + self.masspole
        self.length = 0.5  # lunghezza (metà) del palo
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # intervallo di tempo per l'aggiornamento (in secondi)
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * math.pi / 180  # 12 gradi in radianti

        # Stato: [posizione, velocità, angolo, velocità angolare]
        self.state = None
        self.render_mode = render_mode

        # Se il rendering è abilitato al momento dell'inizializzazione, crea il display
        if self.render_mode:
            pygame.init()
            self.screen_width = 600
            self.screen_height = 400
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("CartPole PPO")
            self.clock = pygame.time.Clock()

    def reset(self):
        # Inizializza lo stato con valori casuali piccoli
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return np.array(self.state)

    def step(self, action):
        # Azione: 0 (forza negativa) o 1 (forza positiva)
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag

        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        # Calcola l'accelerazione intermedia
        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass))
        xacc  = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Aggiorna lo stato utilizzando l'integrazione Euler
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * xacc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * thetaacc
        self.state = (x, x_dot, theta, theta_dot)

        # Reward costante per ogni passo
        reward = 1.0

        # Controllo delle condizioni di terminazione
        done =  x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians

        return np.array(self.state), reward, done, {}

    def render(self):
        if not self.render_mode:
            return

        # Inizializza il rendering se non è stato ancora fatto
        if not hasattr(self, "screen"):
            pygame.init()
            self.screen_width = 600
            self.screen_height = 400
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("CartPole PPO")
            self.clock = pygame.time.Clock()

        # Gestisce gli eventi per assicurare il corretto aggiornamento della finestra
        for event in pygame.event.get():
            # Puoi aggiungere una gestione di pygame.QUIT se vuoi chiudere la finestra
            pass

        # Pulisce lo schermo
        self.screen.fill((255, 255, 255))
        
        # Calcola la posizione del carrello (scalando la posizione x)
        x = self.state[0]
        cartx = int(self.screen_width/2 + (x/self.x_threshold) * (self.screen_width/4))
        carty = int(self.screen_height*0.75)
        
        # Disegna il carrello (rettangolo)
        cart_width = 50
        cart_height = 30
        cart_rect = pygame.Rect(cartx - cart_width//2, carty - cart_height//2, cart_width, cart_height)
        pygame.draw.rect(self.screen, (0, 0, 0), cart_rect)
        
        # Disegna il palo
        pole_length = int(self.length * 200)  # scala per visualizzazione
        theta = self.state[2]
        pole_x = cartx + int(pole_length * math.sin(theta))
        pole_y = carty - int(pole_length * math.cos(theta))
        pygame.draw.line(self.screen, (255, 0, 0), (cartx, carty), (pole_x, pole_y), 5)
        
        pygame.display.flip()
        self.clock.tick(50)  # limita a 50 fps
    
    def close(self):
        if self.render_mode:
            pygame.quit()

    # Metodo opzionale per inizializzare il rendering quando si abilita il rendering in corso di training
    def init_render(self):
        if not hasattr(self, "screen"):
            pygame.init()
            self.screen_width = 600
            self.screen_height = 400
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("CartPole PPO")
            self.clock = pygame.time.Clock()

# Modello Actor-Critic per PPO
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        hidden_size = 64
        # Rete per il policy (attore)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, action_dim)
        )
        # Rete per la stima del valore (critico)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, x):
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        logits, value = self.forward(state)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), value

    def evaluate(self, states, actions):
        logits, values = self.forward(states)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_logprobs = dist.log_prob(actions)
        dist_entropy = dist.entropy()
        return action_logprobs, torch.squeeze(values), dist_entropy

# Calcolo del vantaggio usando GAE
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    # Aggiunge un valore 0 per il terminale
    values = values + [0]
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i+1] * (1 - dones[i]) - values[i]
        gae = delta + gamma * lam * (1 - dones[i]) * gae
        advantages.insert(0, gae)
    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return advantages, returns

# Ciclo principale di training per PPO con GAE
def train():
    env = CartPoleEnv(render_mode=False)
    state_dim = 4
    action_dim = 2
    model = ActorCritic(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    
    # Parametri di training e PPO
    num_episodes = 1000
    max_timesteps = 200
    update_timestep = 2000  # aggiorna il modello ogni 2000 timesteps
    gamma = 0.99
    lam = 0.95
    eps_clip = 0.2
    K_epochs = 4
    batch_size = 64
    
    memory = []  # memorizza (stato, azione, log_prob, reward, done, value)
    timestep = 0
    episode = 0
    
    while episode < num_episodes:
        state = env.reset()
        ep_reward = 0
        for t in range(max_timesteps):
            timestep += 1
            action, log_prob, value = model.act(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward

            memory.append((state, action, log_prob.item(), reward, done, value.item()))
            state = next_state

            # Aggiorna il modello quando si raggiunge il numero prefissato di timesteps
            if timestep % update_timestep == 0:
                # Estrae i dati dalla memoria
                states = torch.FloatTensor([m[0] for m in memory]).to(device)
                actions = torch.LongTensor([m[1] for m in memory]).to(device)
                old_logprobs = torch.FloatTensor([m[2] for m in memory]).to(device)
                rewards = [m[3] for m in memory]
                dones = [m[4] for m in memory]
                values = [m[5] for m in memory]
                
                # Calcola vantaggi e ritorni con GAE
                advantages, returns = compute_gae(rewards, values, dones, gamma, lam)
                advantages = torch.FloatTensor(advantages).to(device)
                returns = torch.FloatTensor(returns).to(device)
                
                # Normalizza i vantaggi
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                # Aggiornamento PPO per K_epochs
                for _ in range(K_epochs):
                    indices = np.arange(len(memory))
                    np.random.shuffle(indices)
                    for i in range(0, len(memory), batch_size):
                        batch_idx = indices[i:i+batch_size]
                        batch_states = states[batch_idx]
                        batch_actions = actions[batch_idx]
                        batch_old_logprobs = old_logprobs[batch_idx]
                        batch_returns = returns[batch_idx]
                        batch_advantages = advantages[batch_idx]
                        
                        logprobs, state_values, dist_entropy = model.evaluate(batch_states, batch_actions)
                        ratios = torch.exp(logprobs - batch_old_logprobs)
                        
                        surr1 = ratios * batch_advantages
                        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * batch_advantages
                        loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(state_values, batch_returns) - 0.01 * dist_entropy.mean()
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                memory = []
                timestep = 0
            
            if done:
                break
        episode += 1
        print(f"Episode {episode} - Reward: {ep_reward}")
        
        # Ogni 50 episodi abilita il rendering per alcune iterazioni
        if episode % 50 == 0:
            env.render_mode = True
            env.init_render()  # Inizializza il rendering se non è già stato fatto
            for _ in range(5):
                state = env.reset()
                for t in range(max_timesteps):
                    env.render()
                    action, _, _ = model.act(state)
                    state, _, done, _ = env.step(action)
                    if done:
                        break
            env.render_mode = False

    env.close()

if __name__ == "__main__":
    train()