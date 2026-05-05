import random
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ============================================================
# 1. EMAIL ENVIRONMENT
# ============================================================
class EmailEnv:
    ACTIONS = {
        0: "répondre maintenant",
        1: "ignorer",
        2: "validation humaine",
        3: "relancer plus tard",
    }

    def __init__(self):
        self.state_dim = 6
        self.action_dim = 4
        self.state = None

    def reset(self):
        self.state = np.array(
            [
                np.random.rand(),  # urgence
                np.random.uniform(-1, 1),  # sentiment
                np.random.rand(),  # importance
                np.random.rand(),  # risque
                np.random.rand(),  # délai
                np.random.rand(),  # confiance
            ],
            dtype=np.float32,
        )
        return self.state, {}

    def estimate_reward(self, state, action):
        u, s, i, r, d, c = state
        if action == 0:
            return u * 0.5 + i * 0.3 + c * 0.2 - r * 0.2
        if action == 1:
            return -u * 0.5 - i * 0.4
        if action == 2:
            return r * 0.5 + i * 0.2
        if action == 3:
            return 0.2 if u < 0.5 else -0.3
        return 0

    def step(self, action):
        next_state = self.state + np.random.normal(0, 0.05, size=6)
        next_state = np.clip(next_state, -1, 1)
        reward = self.estimate_reward(self.state, action)
        self.state = next_state
        return next_state, reward, False, {}


# ============================================================
# 2. WORLD MODEL
# ============================================================
class WorldModel(nn.Module):
    def __init__(self, state_dim=6, action_dim=4):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, state_dim),
        )

    def forward(self, state, action):
        action_onehot = torch.nn.functional.one_hot(
            action.long(), num_classes=self.action_dim
        ).float()
        x = torch.cat([state, action_onehot], dim=-1)
        return self.net(x)


# ============================================================
# 3. MEMORY
# ============================================================
class VectorMemory:
    def __init__(self, dim=6):
        self.dim = dim
        self.index = faiss.IndexFlatL2(6)
        self.storage = []

    def add(self, state, action, next_state, reward):
        state_vec = np.array(state, dtype=np.float32).reshape(1, self.dim)
        self.index.add(state_vec)
        self.storage.append((state, action, next_state, reward))

    def retrieve(self, query_state, k=3):
        if len(self.storage) == 0:
            return []
        query_vec = np.array(query_state, dtype=np.float32).reshape(1, self.dim)
        top_k = min(k, len(self.storage))
        _, indices = self.index.search(query_vec, top_k)
        results = []
        for idx in indices[0]:
            if idx != -1:
                results.append(self.storage[idx])
        return results


# ============================================================
# 4. ORCHESTRATOR
# ============================================================
class Agent:
    def __init__(self):
        self.env = EmailEnv()
        self.model = WorldModel()
        self.memory = VectorMemory(dim=6)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()

    def collect(self):
        state, _ = self.env.reset()
        for _ in range(50):
            action = random.randint(0, 3)
            next_state, reward, _, _ = self.env.step(action)
            self.memory.add(state, action, next_state, reward)
            state = next_state

    def train(self):
        if len(self.memory.storage) < 32:
            return
        batch = random.sample(self.memory.storage, 32)
        states = torch.tensor(
            np.array([b[0] for b in batch]), dtype=torch.float32
        )
        actions = torch.tensor(
            np.array([b[1] for b in batch]), dtype=torch.int64
        )
        next_states = torch.tensor(
            np.array([b[2] for b in batch]), dtype=torch.float32
        )
        pred = self.model(states, actions)
        loss = self.loss_fn(pred, next_states)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def decide(self, state):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        memories = self.memory.retrieve(state, k=3)
        print("Similar memories:")
        for memory in memories:
            _, mem_action, _, mem_reward = memory
            print(
                f"- action: {self.env.ACTIONS[mem_action]} | reward: {mem_reward:.2f}"
            )
        results = []
        for a in range(4):
            action_t = torch.tensor([a])
            self.model(state_t, action_t)
            estimated = self.env.estimate_reward(state, a)
            same_action_rewards = [m[3] for m in memories if m[1] == a]
            if same_action_rewards:
                score = estimated + (sum(same_action_rewards) / len(same_action_rewards)) * 0.3
            else:
                score = estimated
            results.append((a, score))
        best = max(results, key=lambda x: x[1])
        return best


# ============================================================
# 5. MAIN
# ============================================================
if __name__ == "__main__":
    agent = Agent()
    for epoch in range(30):
        agent.collect()
        loss = agent.train()
        state, _ = agent.env.reset()
        print(f"Epoch {epoch}")
        print(f"State: {np.round(state, 2)}")
        action, score = agent.decide(state)
        print(f"Decision: {agent.env.ACTIONS[action]}")
        print(f"Score: {score:.2f}")
        print(f"Loss: {loss}")
        print(f"Memory size: {len(agent.memory.storage)}")
