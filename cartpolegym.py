import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time

# Configurazione del dispositivo (GPU se disponibile)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Definizione del modello Attore-Critico ------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super(ActorCritic, self).__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh()
        )
        self.policy_head = nn.Linear(64, act_dim)
        self.value_head = nn.Linear(64, 1)
    
    def forward(self, x):
        x = self.shared(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value
    
    def act(self, x):
        logits, value = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action, dist.log_prob(action), value

    def evaluate_actions(self, x, action):
        logits, value = self.forward(x)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, torch.squeeze(value), dist_entropy

# ------------------ Calcolo del GAE ------------------
def compute_gae(rewards, masks, values, next_value, gamma, lam):
    """
    Calcola gli advantage e i returns usando Generalized Advantage Estimation (GAE).
    - rewards: tensor di shape (num_steps, num_envs)
    - masks: tensor di shape (num_steps, num_envs), 1 se l'ambiente non è terminale, 0 altrimenti
    - values: tensor di shape (num_steps, num_envs)
    - next_value: tensor di shape (num_envs,)
    """
    num_steps = rewards.size(0)
    advantages = torch.zeros_like(rewards).to(device)
    gae = 0
    for step in reversed(range(num_steps)):
        next_val = next_value if step == num_steps - 1 else values[step + 1]
        delta = rewards[step] + gamma * next_val * masks[step] - values[step]
        gae = delta + gamma * lam * masks[step] * gae
        advantages[step] = gae
    returns = advantages + values
    return advantages, returns


class CustomTrackCartPole(gym.CartPoleEnv):
    metadata = {

        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }
    
    def __init__(self, render_mode=None):
        # Pass the render_mode to the gym environment
        self.env = gym.make('CartPole-v1', render_mode=None)
        
        self.env = gym.wrappers.Autoreset(self.env)
        
        # Raddoppia la soglia x
        self.x_threshold = 2.4*10  # doppio dell'originale (2.4)
        self.env.unwrapped.theta_threshold_radians = np.pi * 3  # 2pi
        
        # Mantieni lo stesso spazio di azione
        self.action_space = self.env.action_space
        
        # Ridefinisci lo spazio di osservazione
        high = np.array(
            [
                self.x_threshold,
                np.finfo(np.float32).max,
                self.env.unwrapped.theta_threshold_radians,
                np.finfo(np.float32).max
            ],
            dtype=np.float32
        )
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.render_mode = render_mode
        self.screen = None
        self.screen_width = 1200  # più largo per ospitare la rotaia più lunga
        self.screen_height = 400
        self.clock = None
        
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return observation, info
        
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        # Estrai le componenti dell'osservazione
        x = observation[0]
        theta = observation[2]  # l'angolo dell'asta
        
        # Terminazione basata SOLO sulla posizione x (ignora l'angolo)
        done_x = bool(
            x < -self.x_threshold
            or x > self.x_threshold
        )
        
        # Sovrascrivi la condizione terminated dell'ambiente originale
        # Termina solo se fuori dai limiti x
        if terminated:
            # Verifica se è stato terminato per l'angolo
            if abs(theta) > self.env.unwrapped.theta_threshold_radians and not done_x:
                # Ignora la terminazione se è dovuta solo all'angolo
                terminated = False
        
        # Assicurati di terminare se il carrello è fuori dai limiti estesi
        if done_x:
            terminated = True
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode is None:
            return
            
        import pygame
        from pygame import gfxdraw
        
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:  # rgb_array
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
            
        if self.clock is None:
            self.clock = pygame.time.Clock()
            
        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))
        
        # Ottieni lo stato dall'ambiente originale
        x, x_dot, theta, theta_dot = self.env.unwrapped.state
        
        # Scala per la visualizzazione
        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        cart_y = 200  # posizione verticale del carrello
        pole_length = scale * (2 * self.env.unwrapped.length) * 5
        
        # Disegna la rotaia
        pygame.draw.line(
            self.surf,
            (0, 0, 0),
            (0, cart_y),
            (self.screen_width, cart_y),
            1
        )
        
        # Disegna il carrello
        cart_x = x * scale + self.screen_width / 2.0
        cart_width = 40
        cart_height = 30
        
        rect = pygame.Rect(
            cart_x - cart_width / 2,
            cart_y - cart_height / 2,
            cart_width,
            cart_height
        )
        pygame.draw.rect(self.surf, (0, 0, 0), rect)
        
        # Disegna il palo
        pole_x = np.sin(theta) * pole_length
        pole_y = np.cos(theta) * pole_length
        
        pygame.draw.line(
            self.surf,
            (0, 0, 255),
            (cart_x, cart_y),
            (cart_x + pole_x, cart_y + pole_y),
            6
        )
        
        # Aggiorna lo schermo
        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
            
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
    
    def close(self):
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None



# ------------------ Factory per la creazione degli ambienti ------------------
def make_env():
    def _init():
        # return gym.make(env_name)
        return CustomTrackCartPole()
    return _init

# ------------------ Iperparametri ------------------
num_envs = 20            # Numero di ambienti paralleli
num_steps = 128         # Lunghezza dell'orizzonte (rollout) per ogni update
num_updates = 1000      # Numero di update (batch)
ppo_epochs = 4          # Numero di passaggi di ottimizzazione per batch
mini_batch_size = 64    # Dimensione dei mini-batch
gamma = 0.99            # Fattore di sconto
gae_lambda = 0.95       # Parametro lambda per GAE
clip_epsilon = 0.2      # Clipping per l'aggiornamento PPO
value_loss_coef = 0.5   # Coefficiente per la loss di stima del valore
entropy_coef = 0.01     # Coefficiente per il bonus entropico
learning_rate = 3e-4
sim_interval = 10      # Ogni quanti update visualizzare un episodio

# ------------------ Creazione degli ambienti vettorializzati ------------------
env_fns = [make_env() for _ in range(num_envs)]
vec_env = AsyncVectorEnv(env_fns)
obs, infos = vec_env.reset()  # Stato iniziale per tutti gli ambienti

# Estrazione delle dimensioni degli spazi osservazione/azione
obs_dim = vec_env.single_observation_space.shape[0]
act_dim = vec_env.single_action_space.n

# ------------------ Inizializzazione del modello e dell'ottimizzatore ------------------
model = ActorCritic(obs_dim, act_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ------------------ Ciclo di Training con PPO e GAE ------------------
for update in range(1, num_updates + 1):
    # Buffer per memorizzare i dati del rollout
    obs_buffer = []
    actions_buffer = []
    logprobs_buffer = []
    rewards_buffer = []
    masks_buffer = []
    values_buffer = []
    
    # Conversione dello stato iniziale in tensore
    obs_tensor = torch.FloatTensor(obs).to(device)
    
    # Raccolta del rollout per num_steps per ogni ambiente
    for step in range(num_steps):
        with torch.no_grad():
            action, log_prob, value = model.act(obs_tensor)
        obs_buffer.append(obs_tensor)
        actions_buffer.append(action)
        logprobs_buffer.append(log_prob)
        values_buffer.append(value.squeeze())
        
        # Esecuzione dello step negli ambienti paralleli
        actions_cpu = action.cpu().numpy()
        next_obs, rewards, terminated, truncated, infos = vec_env.step(actions_cpu)
        # Calcolo della maschera: 1 se l'episodio continua, 0 altrimenti
        masks = 1 - (terminated | truncated).astype(np.float32)
        rewards_buffer.append(torch.FloatTensor(rewards).to(device))
        masks_buffer.append(torch.FloatTensor(masks).to(device))
        
        obs_tensor = torch.FloatTensor(next_obs).to(device)
    
    # Calcolo del valore per l'ultimo stato
    with torch.no_grad():
        _, next_value = model.forward(obs_tensor)
    next_value = next_value.squeeze()
    
    # Converte le liste in tensori e li rimodella
    obs_buffer = torch.stack(obs_buffer)           # (num_steps, num_envs, obs_dim)
    actions_buffer = torch.stack(actions_buffer)     # (num_steps, num_envs)
    logprobs_buffer = torch.stack(logprobs_buffer)     # (num_steps, num_envs)
    values_buffer = torch.stack(values_buffer)         # (num_steps, num_envs)
    rewards_buffer = torch.stack(rewards_buffer)       # (num_steps, num_envs)
    masks_buffer = torch.stack(masks_buffer)           # (num_steps, num_envs)
    
    # Calcolo degli advantage e dei returns
    advantages, returns = compute_gae(rewards_buffer, masks_buffer, values_buffer, next_value, gamma, gae_lambda)
    
    # Rimodelliamo i dati in batch: (num_steps * num_envs, ...)
    batch_obs = obs_buffer.view(-1, obs_dim)
    batch_actions = actions_buffer.view(-1)
    batch_logprobs = logprobs_buffer.view(-1)
    batch_returns = returns.view(-1)
    batch_advantages = advantages.view(-1)
    
    # Normalizziamo gli advantage
    batch_advantages = (batch_advantages - batch_advantages.mean()) / (batch_advantages.std() + 1e-8)
    
    # Aggiornamento PPO su mini-batch per ppo_epochs
    dataset_size = batch_obs.size(0)
    indices = np.arange(dataset_size)
    for epoch in range(ppo_epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, mini_batch_size):
            end = start + mini_batch_size
            mb_idx = indices[start:end]
            
            mb_obs = batch_obs[mb_idx]
            mb_actions = batch_actions[mb_idx]
            mb_logprobs_old = batch_logprobs[mb_idx]
            mb_returns = batch_returns[mb_idx]
            mb_advantages = batch_advantages[mb_idx]
            
            new_logprobs, values, entropy = model.evaluate_actions(mb_obs, mb_actions)
            ratio = torch.exp(new_logprobs - mb_logprobs_old)
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values, mb_returns)
            loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    if update % 10 == 0:
        print(f"Update {update} - Loss: {loss.item():.3f} - Avg Return: {batch_returns.mean().item():.3f}")
    
    # Visualizzazione di un episodio ogni sim_interval update
    if update % sim_interval == 0:
        print("Visualizzazione episodio con modello attuale...")
        # Usa il tuo ambiente personalizzato invece di quello standard
        sim_env = CustomTrackCartPole(render_mode="human")

        obs_sim, info = sim_env.reset()
        done = False
        obs_sim_tensor = torch.FloatTensor(obs_sim).to(device)
        while not done:
            sim_env.render()
            with torch.no_grad():
                logits, _ = model.forward(obs_sim_tensor.unsqueeze(0))
                probs = torch.softmax(logits, dim=-1)
                action = torch.argmax(probs, dim=-1).item()
            obs_sim, sim_reward, terminated, truncated, sim_info = sim_env.step(action)
            done = terminated or truncated
            obs_sim_tensor = torch.FloatTensor(obs_sim).to(device)
            time.sleep(0.02)
        sim_env.close()

# Chiusura degli ambienti vettorializzati al termine del training
vec_env.close()
print("Training completato!")
