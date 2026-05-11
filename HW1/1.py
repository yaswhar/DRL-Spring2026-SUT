# %% [markdown]
#   # Deep Q-Network (DQN)
# 
#  * Full Name: Yashar Zafari
# 
#  * Student Number: 404210253

# %%
import os, random, cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque, namedtuple
import matplotlib.pyplot as plt
import gymnasium as gym
from IPython.display import HTML
from base64 import b64encode
import mediapy as media


# %%
# ========================  Configuration  ========================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %%
def show_video(video_path, width=480):
    with open(video_path, "rb") as f:
        video_data = f.read()
    encoded = b64encode(video_data).decode("utf-8")
    html = f'''
        <video width="{width}" controls>
            <source src="data:video/mp4;base64,{encoded}" type="video/mp4">
        </video>
    '''
    return HTML(html)

def save_agent_video(env, agent, filename="agent_demo.mp4"):
    frames = []
    state, _ = env.reset()
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        frames.append(env.render())
        state = next_state

    height, width, layers = frames[0].shape
    out = cv2.VideoWriter(
        filename, cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height)
    )
    for f in frames:
        out.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    out.release()


# %%
# =================  Environment Check (LunarLander-v3)  =========
env = gym.make('LunarLander-v2', render_mode='rgb_array')
nA = env.action_space.n
nS = env.observation_space.shape[0]
print(f'Action space: {nA}, Observation space: {nS}')

class TestAgent:
    def __init__(self, nA):
        self.nA = nA
    def select_action(self, s, training=False):
        return random.randint(0, self.nA - 1)

# Create demo video if it doesn't exist
test_video_path = 'initial test/agent_test.mp4'
if not os.path.exists(test_video_path):
    os.makedirs(os.path.dirname(test_video_path), exist_ok=True)
    test_env = gym.make('LunarLander-v2', render_mode='rgb_array')
    save_agent_video(test_env, TestAgent(nA), filename=test_video_path)
    test_env.close()
show_video(test_video_path)


# %%
# ======================  Data Structures  =======================
Transition = namedtuple('Transition', ['state', 'action', 'reward', 'next_state', 'done'])

class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# %%
# ======================  Neural Networks  =======================
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.network(x)

class DuelingQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU()
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        feat = self.features(x)
        value = self.value_stream(feat)
        advantage = self.advantage_stream(feat)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


# %%
# ==================  Base DQN Agent (refactored) ================
class BaseDQNAgent:
    def __init__(self, state_dim, action_dim, device,
                 q_network=None, target_network=None,
                 gamma=0.99, epsilon=0.6, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=64, target_update_freq=2000, lr=0.001):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.action_dim = action_dim
        self.device = device

        self.q_network = q_network.to(device)
        self.target_network = target_network.to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = ReplayBuffer()
        self.steps = 0

    def select_action(self, state, training=True):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            return self.q_network(state_t).argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def _prepare_batch(self, batch):
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([[t.action] for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([[t.reward] for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([[t.done] for t in batch], dtype=torch.float32, device=self.device)
        return states, actions, rewards, next_states, dones

    def update(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self._prepare_batch(batch)

        current_q = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            # Vanilla DQN: max over target network
            next_q = self.target_network(next_states).max(1, keepdim=True).values
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return loss.item()


# %%
# ======================  Double DQN Agent =======================
class DDQNAgent(BaseDQNAgent):
    def __init__(self, state_dim, action_dim, device,
                 q_network, target_network, **kwargs):
        super().__init__(state_dim, action_dim, device,
                         q_network=q_network, target_network=target_network, **kwargs)

    def update(self):
        if len(self.memory) < self.batch_size:
            return None
        batch = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones = self._prepare_batch(batch)

        current_q = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            # Double DQN: online selects, target evaluates
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = F.mse_loss(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return loss.item()


# %%
# =====================  Dueling DQN Agent =======================
class DuelingDQNAgent(DDQNAgent):   # Inherits Double DQN update logic, just uses Dueling networks
    def __init__(self, state_dim, action_dim, device, **kwargs):
        q_net = DuelingQNetwork(state_dim, action_dim)
        target_net = DuelingQNetwork(state_dim, action_dim)
        super().__init__(state_dim, action_dim, device,
                         q_network=q_net, target_network=target_net, **kwargs)


# %%
# ========================  PER (SumTree) ========================
class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float32)
        self.data = np.empty(capacity, dtype=object)
        self.write_idx = 0
        self.size = 0

    def add(self, priority, transition):
        idx = self.write_idx + self.capacity - 1
        self.data[self.write_idx] = transition
        self.update(idx, priority)
        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def sample(self, batch_size):
        """Return (batch, priorities, indices) – guaranteed no None transitions."""
        if self.size == 0:
            raise RuntimeError("SumTree is empty, cannot sample")

        batch = []
        indices = []
        priorities = []

        total_priority = self.tree[0]   # root of the tree array
        if total_priority == 0.0:
            raise RuntimeError("SumTree total priority is zero")

        segment = total_priority / batch_size

        for i in range(batch_size):
            low = segment * i
            high = segment * (i + 1)

            # Try up to 50 times to hit a valid leaf
            for _ in range(50):
                r = random.uniform(low, high)
                idx = self._find_leaf(0, r)
                data_idx = idx - self.capacity + 1
                if self.data[data_idx] is not None:
                    break
                # Expand range slightly if stuck
                low = max(0.0, low - 0.01 * total_priority)
                high = min(total_priority, high + 0.01 * total_priority)
            else:
                # Fallback: pick any valid leaf
                valid = [j for j, d in enumerate(self.data) if d is not None]
                data_idx = random.choice(valid)
                idx = data_idx + self.capacity - 1

            batch.append(self.data[data_idx])
            priorities.append(self.tree[idx])
            indices.append(idx)

        return batch, priorities, indices

    def _find_leaf(self, idx, value):
        while True:
            left = 2 * idx + 1
            right = left + 1
            if left >= len(self.tree):
                return idx
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right

    def __len__(self):
        return self.size

class PrioritizedReplayBuffer:
    def __init__(self, capacity=50000, alpha=0.6, beta=0.4, beta_increment=5e-4, device='cpu'):
        self.tree = SumTree(capacity)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = 1e-5
        self.max_priority = 1.0                # will be updated automatically
        self.device = device

    def push(self, transition):
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size):
        """Return (states, actions, rewards, ...) tensors on the correct device."""
        if len(self.tree) < batch_size:
            raise RuntimeError("Not enough transitions in buffer to sample a batch")

        # Sample from SumTree; guaranteed no None transitions now
        batch, priorities, indices = self.tree.sample(batch_size)

        # Convert to numpy priority array
        priorities = np.array(priorities, dtype=np.float32)
        total_priority = self.tree.tree[0]   # root priority
        probs = priorities / total_priority

        weights = (self.tree.size * probs) ** (-self.beta)
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights /= weights.max()   # stable normalization

        # Build tensors
        states = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=self.device)
        actions = torch.tensor([[t.action] for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.tensor([[t.reward] for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=self.device)
        dones = torch.tensor([[t.done] for t in batch], dtype=torch.float32, device=self.device)
        weights = torch.tensor(weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, td_errors):
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self):
        return len(self.tree)


# %%
# ======================  PER DQN Agent ==========================
class PERDQNAgent:
    def __init__(self, state_dim, action_dim, device, gamma=0.99,
                 epsilon=0.6, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=64, target_update_freq=2000, lr=0.001):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.action_dim = action_dim
        self.device = device

        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = PrioritizedReplayBuffer(capacity=50000, alpha=0.6, beta=0.4,
                                              beta_increment=1e-4, device=device)
        self.steps = 0

    def select_action(self, state, training=True):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            return self.q_network(state_t).argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(Transition(state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size)

        current_q = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            # Double DQN target for PER
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # *** FIX: Huber loss (recommended by PER paper) ***
        loss_per_element = F.smooth_l1_loss(current_q, target_q, reduction='none')
        loss = (weights * loss_per_element).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # *** FIX: use squeeze(1) to avoid 0-d case ***
        td_errors = (current_q - target_q).detach().abs().squeeze(1).cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return loss.item()


# %%
# ==================  Combined (Half‑Rainbow) Agent ==============
class CombinedDQNAgent:
    def __init__(self, state_dim, action_dim, device, gamma=0.99,
                 epsilon=0.6, epsilon_min=0.01, epsilon_decay=0.995,   # <-- aligned to 0.995
                 batch_size=64, target_update_freq=2000, lr=0.001):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.action_dim = action_dim
        self.device = device

        self.q_network = DuelingQNetwork(state_dim, action_dim).to(device)
        self.target_network = DuelingQNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        # ** FIX: beta_increment = 1e-4 **
        self.memory = PrioritizedReplayBuffer(capacity=50000, alpha=0.6, beta=0.4,
                                              beta_increment=1e-4, device=device)
        self.steps = 0

    def select_action(self, state, training=True):
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        with torch.no_grad():
            return self.q_network(state_t).argmax(dim=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(Transition(state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones, weights, indices = self.memory.sample(self.batch_size)

        current_q = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            # Double DQN + Dueling
            next_actions = self.q_network(next_states).argmax(1, keepdim=True)
            next_q = self.target_network(next_states).gather(1, next_actions)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        # *** Huber loss ***
        loss_per_element = F.smooth_l1_loss(current_q, target_q, reduction='none')
        loss = (weights * loss_per_element).mean()

        self.optimizer.zero_grad()
        loss.backward()

        # *** Gradient clipping (uncommented) ***
        nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
        self.optimizer.step()

        td_errors = (current_q - target_q).detach().abs().squeeze(1).cpu().numpy()
        self.memory.update_priorities(indices, td_errors)

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return loss.item()


# %%
# =======================  Training Loop =========================
def train_dqn(env, agent, num_episodes=500, max_steps=1000):
    scores = []
    losses = []
    for episode in range(num_episodes):
        state, _ = env.reset(seed=SEED + episode)
        score = 0
        ep_loss = 0.0
        n_updates = 0
        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.store_transition(state, action, reward, next_state, done)
            loss_val = agent.update()
            if loss_val is not None:
                ep_loss += loss_val
                n_updates += 1

            score += reward
            state = next_state
            if done:
                break

        scores.append(score)
        losses.append(ep_loss / n_updates if n_updates > 0 else 0.0)

        if (episode + 1) % 50 == 0:
            avg_score = np.mean(scores[-50:])
            print(f"Ep {episode+1:3d} | Score: {score:6.1f} | Avg50: {avg_score:6.1f} | ε: {agent.epsilon:.3f} | Loss: {losses[-1]:.4f}")

    return scores, losses


# %%
# ===================  Run All Experiments =======================
env = gym.make('LunarLander-v2')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

def run_experiment(name, agent_class, agent_kwargs, env):
    print("="*60)
    print(f"Training {name} on LunarLander-v2")
    print("="*60)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(SEED)
    agent = agent_class(**agent_kwargs)
    scores, losses = train_dqn(env, agent)
    return agent, scores, losses

# DQN
dqn_agent, dqn_scores, dqn_losses = run_experiment(
    "DQN", BaseDQNAgent,
    dict(state_dim=state_dim, action_dim=action_dim, device=device,
         q_network=QNetwork(state_dim, action_dim),
         target_network=QNetwork(state_dim, action_dim)),
    env
)
# DDQN
ddqn_agent, ddqn_scores, ddqn_losses = run_experiment(
    "Double DQN", DDQNAgent,
    dict(state_dim=state_dim, action_dim=action_dim, device=device,
         q_network=QNetwork(state_dim, action_dim),
         target_network=QNetwork(state_dim, action_dim)),
    env
)
# Dueling DQN (inherits Double DQN update)
dueling_agent, dueling_scores, dueling_losses = run_experiment(
    "Dueling DQN", DuelingDQNAgent,
    dict(state_dim=state_dim, action_dim=action_dim, device=device),
    env
)
# PER DQN
per_agent, per_scores, per_losses = run_experiment(
    "PER DQN", PERDQNAgent,
    dict(state_dim=state_dim, action_dim=action_dim, device=device),
    env
)
# Combined (Half‑Rainbow)
combined_agent, combined_scores, combined_losses = run_experiment(
    "Combined (Double+Dueling+PER)",
    CombinedDQNAgent,
    dict(state_dim=state_dim, action_dim=action_dim, device=device),
    env
)


# %%
# =====================  Visualisation  ==========================
window = 50
scores_dict = {
    'DQN': dqn_scores,
    'Double DQN': ddqn_scores,
    'Dueling DQN': dueling_scores,
    'PER DQN': per_scores,
    'Combined': combined_scores,
}
colors = ['blue', 'red', 'green', 'purple', 'orange']

# Moving averages
ma_dict = {name: np.convolve(scores, np.ones(window)/window, mode='valid')
           for name, scores in scores_dict.items()}

# Figure 1: Moving average of all methods + bar chart
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
for (name, ma), col in zip(ma_dict.items(), colors):
    ax1.plot(ma, color=col, linewidth=2, label=name)
ax1.axhline(200, color='black', linestyle='--', label='Target (200)')
ax1.set_title('Moving Average (50) – All Methods')
ax1.set_xlabel('Episode'); ax1.set_ylabel('Score')
ax1.legend(); ax1.grid(True, alpha=0.3)

# Bar chart of final 50 episodes
averages = [np.mean(scores[-50:]) for scores in scores_dict.values()]
bars = ax2.bar(scores_dict.keys(), averages, color=colors, alpha=0.7)
ax2.axhline(200, color='black', linestyle='--', label='Target')
for bar, avg in zip(bars, averages):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+2,
             f'{avg:.1f}', ha='center', fontweight='bold')
ax2.set_title('Average Score (Last 50 Episodes)')
ax2.set_ylabel('Score')
ax2.legend()

plt.tight_layout()
plt.show()

# Figure 2: Pairwise comparisons
pairs = [('DQN', 'Double DQN'), ('Double DQN', 'Dueling DQN'),
         ('Dueling DQN', 'PER DQN'), ('PER DQN', 'Combined')]
fig2, axes = plt.subplots(2, 2, figsize=(12, 9))
for ax, (a, b) in zip(axes.flatten(), pairs):
    ax.plot(ma_dict[a], label=a, linewidth=2)
    ax.plot(ma_dict[b], label=b, linewidth=2)
    ax.axhline(200, color='black', linestyle='--')
    ax.set_title(f'{a} vs {b}')
    ax.legend(); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%
# Final summary
print("\n" + "="*60)
print("FINAL SUMMARY (Last 50 Episodes)")
print("="*60)
for name, scores in scores_dict.items():
    print(f"{name:<20}: {np.mean(scores[-50:]):.1f}")
print("="*60)

# Save and show video of the combined agent
env_render = gym.make('LunarLander-v2', render_mode='rgb_array')
video_file = save_agent_video(env_render, combined_agent, 'halfrainbow_v3.mp4')
print("Video saved:", video_file)
show_video(video_file)


