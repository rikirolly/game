import gymnasium as gym
import numpy as np
from gymnasium.envs.classic_control.cartpole import CartPoleEnv
from stable_baselines3 import PPO

class FreeRotationCartPoleEnv(CartPoleEnv):
    def __init__(self):
        super().__init__()
        # Imposta la soglia angolare a infinito per permettere la rotazione completa dell'asta
        self.theta_threshold_radians = np.inf

    def reset(self, seed=None, options=None):
        # Chiama il reset base per inizializzare il generatore di numeri casuali
        super().reset(seed=seed)
        # Stato iniziale: [posizione, velocità, angolo, velocità angolare]
        # L'asta viene inizialmente posizionata in posizione "inversa" (θ = π) con un piccolo rumore
        initial_state = np.array([0.0, 0.0, np.pi, 0.0], dtype=np.float32)
        if seed is None:
            self.np_random, seed = gym.utils.seeding.np_random()
        noise = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state = initial_state + noise
        return np.array(self.state, dtype=np.float32), {}

# Crea l'ambiente personalizzato
env = FreeRotationCartPoleEnv()

# (Opzionale) Verifica che l'ambiente sia conforme agli standard Gymnasium
# from gymnasium.utils.env_checker import check_env
# check_env(env, warn=True)

# Crea il modello PPO per addestrare l'agente sull'ambiente personalizzato
model = PPO("MlpPolicy", env, verbose=1)

# Avvia il training per 100.000 timesteps (puoi regolare questo valore)
model.learn(total_timesteps=100000)

# Salva il modello addestrato
model.save("ppo_free_rotation_cartpole")
