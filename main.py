import os
import random

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faiss
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# =========================
# ENV SIMULÉ
# =========================


class EmailEnv:
    ACTIONS = ["répondre", "ignorer", "valider", "relancer"]
    action_dim = 4

    def reset(self):
        return np.random.rand(4), {}

    def step(self, action):
        next_state = np.random.rand(4)
        reward = np.random.randn() * 0.5 + (1 if action == 0 else -0.2)
        done = False
        return next_state, reward, done, {}

    def real_outcome(self, state, action):
        next_state = np.random.rand(4)
        reward = np.random.randn() * 0.5 + (1 if action == 0 else -0.2)
        return next_state, reward


# =========================
# WORLD MODEL
# =========================


class WorldModel(nn.Module):
    def __init__(self, state_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim + 1, 32),
            nn.ReLU(),
            nn.Linear(32, state_dim),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.model(x)


# =========================
# FAISS MEMORY (STM)
# =========================


class VectorMemory:
    def __init__(self, dim):
        self.dim = dim
        self.index = faiss.IndexFlatL2(dim)
        self.storage = []

    def add(self, state, action, next_state, reward):
        vec = np.array(state, dtype=np.float32)
        self.index.add(vec.reshape(1, -1))
        self.storage.append((state, action, next_state, reward))

    def retrieve(self, state, k=5):
        if len(self.storage) == 0:
            return []
        vec = np.array(state, dtype=np.float32)
        kk = min(k, len(self.storage))
        _, I = self.index.search(vec.reshape(1, -1), kk)
        return [self.storage[i] for i in I[0] if i != -1]

    def retrieve_by_action(self, state, action, k=5):
        memories = self.retrieve(state, k=10)
        return [m for m in memories if m[1] == action][:k]


# =========================
# STRATEGIC MEMORY (LTM)
# =========================


class StrategicMemory:
    def __init__(self):
        self.stats = {}
        self.rules = []

    def update_stats(self, action, reward):
        if action not in self.stats:
            self.stats[action] = {"count": 0, "sum": 0}
        self.stats[action]["count"] += 1
        self.stats[action]["sum"] += reward

    def extract_context(self, state):
        s = np.asarray(state).reshape(-1)
        return {
            "importance_high": float(s[2]) > 0.6,
            "urgency_high": float(s[0]) > 0.6,
            "risk_high": float(s[3]) > 0.6,
        }

    def context_similarity(self, ctx1, ctx2):
        score = 0
        total = len(ctx1)
        if total == 0:
            return 1.0
        for key in ctx1:
            if ctx1[key] == ctx2[key]:
                score += 1
        return score / total

    def maybe_create_rule(self, action, state):
        stat = self.stats[action]
        avg = stat["sum"] / stat["count"]
        if stat["count"] < 5:
            return None
        context = self.extract_context(state)
        if avg < -0.3:
            return {
                "action": action,
                "rule": "avoid",
                "confidence": min(1.0, abs(avg)),
                "context": context,
            }
        if avg > 0.5:
            return {
                "action": action,
                "rule": "prefer",
                "confidence": min(1.0, avg),
                "context": context,
            }
        return None

    def store(self, lesson):
        for r in self.rules:
            if (
                r["action"] == lesson["action"]
                and r["rule"] == lesson["rule"]
                and r["context"] == lesson["context"]
            ):
                return
        self.rules.append(lesson)

    def match(self, state):
        current_context = self.extract_context(state)
        matched = []
        for rule in self.rules:
            sim = self.context_similarity(rule["context"], current_context)
            if sim > 0.66:
                weighted_confidence = rule["confidence"] * sim
                matched.append(
                    {
                        "action": rule["action"],
                        "rule": rule["rule"],
                        "confidence": weighted_confidence,
                        "similarity": sim,
                    }
                )
        return matched


# =========================
# AGENT
# =========================


class Agent:
    def __init__(self):
        self.env = EmailEnv()
        self.memory = VectorMemory(dim=4)
        self.long_memory = StrategicMemory()
        self.world_model = WorldModel(4)
        self.optimizer = optim.Adam(self.world_model.parameters(), lr=0.01)
        self.criterion = nn.MSELoss()
        self.epsilon = 1.0
        self.epsilon_decay = 0.95
        self.epsilon_min = 0.1

    def estimate_uncertainty(self, state, action):
        memories = self.memory.retrieve_by_action(state, action)
        if len(memories) < 2:
            return 1.0
        rewards = [m[3] for m in memories]
        return float(np.std(rewards))

    def rollout(self, state, action, horizon=3):
        total_score = 0
        current_state = np.asarray(state, dtype=np.float32).copy()
        for _ in range(horizon):
            state_t = torch.tensor(current_state, dtype=torch.float32).unsqueeze(
                0
            )
            action_t = torch.tensor([[action]], dtype=torch.float32)
            with torch.no_grad():
                next_state = self.world_model(state_t, action_t).numpy()[0]
            memories = self.memory.retrieve(next_state)
            memory_reward = (
                np.mean([m[3] for m in memories]) if memories else 0
            )
            rules = self.long_memory.match(current_state)
            for r in rules:
                print(
                    f"Rule → action {r['action']} | {r['rule']} | "
                    f"conf={round(r['confidence'], 2)} | "
                    f"sim={round(r['similarity'], 2)}"
                )
            penalty = 0
            for r in rules:
                if r["rule"] == "avoid":
                    penalty -= r["confidence"]
                elif r["rule"] == "prefer":
                    penalty += r["confidence"]
            score = memory_reward + penalty
            total_score += score
            current_state = next_state.astype(np.float32)
        return total_score, current_state

    def select_action(self, state):
        explore = random.random() < self.epsilon
        scores = []
        for action in range(self.env.action_dim):
            uncertainty = self.estimate_uncertainty(state, action)
            rollout_score, final_state = self.rollout(state, action)
            rollout_norm = np.tanh(rollout_score)
            score = uncertainty + 0.15 * rollout_norm
            scores.append(
                (
                    action,
                    score,
                    rollout_score,
                    uncertainty,
                    final_state,
                )
            )
        scores.sort(key=lambda x: x[1], reverse=True)
        if explore:
            chosen = scores[0]
            mode = "exploration intelligente"
        else:
            chosen = max(scores, key=lambda x: x[2])
            mode = "exploitation"
        return chosen, mode

    def train_model(self, state, action, next_state):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_t = torch.tensor([[action]], dtype=torch.float32)
        next_t = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        pred = self.world_model(state_t, action_t)
        loss = self.criterion(pred, next_t)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def apply_feedback(self, state, action):
        next_state, reward = self.env.real_outcome(state, action)
        self.memory.add(state, action, next_state, reward)
        self.long_memory.update_stats(action, reward)
        lesson = self.long_memory.maybe_create_rule(action, state)
        if lesson:
            self.long_memory.store(lesson)
        return next_state, reward


# =========================
# MAIN LOOP
# =========================


if __name__ == "__main__":
    agent = Agent()
    errors = []
    for epoch in range(30):
        state, _ = agent.env.reset()
        action_info, mode = agent.select_action(state)
        action, _, rollout_score, uncertainty, _ = action_info
        next_state, reward = agent.apply_feedback(state, action)
        _ = agent.train_model(state, action, next_state)
        reward_f = float(reward)
        rollout_score_f = float(rollout_score)
        error = abs(rollout_score_f - reward_f)
        errors.append(error)
        if len(errors) > 50:
            errors.pop(0)
        avg_error = sum(errors) / len(errors)
        agent.epsilon = max(
            agent.epsilon * agent.epsilon_decay, agent.epsilon_min
        )
        print(
            {
                "epoch": epoch,
                "mode": mode,
                "action": agent.env.ACTIONS[action],
                "uncertainty": round(float(uncertainty), 3),
                "rollout_score": round(rollout_score_f, 3),
                "real_reward": round(reward_f, 3),
                "prediction_error": round(error, 3),
                "avg_prediction_error": round(avg_error, 3),
                "memory_size": len(agent.memory.storage),
                "rules_count": len(agent.long_memory.rules),
            }
        )
