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
# 4. GOAL SYSTEM
# ============================================================
class GoalSystem:
    def __init__(self):
        self.weights = {
            "client_satisfaction": 0.25,
            "risk_reduction": 0.25,
            "urgency_control": 0.25,
            "confidence_growth": 0.25,
        }
        self.normalize_weights()

    def normalize_weights(self):
        total = sum(self.weights.values())
        for key in self.weights:
            self.weights[key] = self.weights[key] / total

    def evaluate_state_value(self, state):
        vec = np.asarray(state, dtype=np.float64).reshape(-1)
        u = float(vec[0])
        sentiment = float(vec[1])
        importance = float(vec[2])
        risk = float(vec[3])
        confidence = float(vec[5])
        signals = {
            "client_satisfaction": ((sentiment + 1) / 2) * importance,
            "risk_reduction": 1 - risk,
            "urgency_control": 1 - u,
            "confidence_growth": confidence,
        }
        return float(sum(self.weights[k] * signals[k] for k in self.weights))

    def update_weights(self, state, reward):
        vec = np.asarray(state, dtype=np.float64).reshape(-1)
        urgency = float(vec[0])
        sentiment = float(vec[1])
        importance = float(vec[2])
        risk = float(vec[3])
        confidence = float(vec[5])

        signals = {
            "client_satisfaction": ((sentiment + 1) / 2) * importance,
            "risk_reduction": 1 - risk,
            "urgency_control": 1 - urgency,
            "confidence_growth": confidence,
        }

        learning_rate = 0.03

        for key, signal in signals.items():
            if reward > 0:
                self.weights[key] += learning_rate * signal
            else:
                self.weights[key] -= learning_rate * signal

            self.weights[key] = max(0.05, min(0.70, self.weights[key]))

        self.normalize_weights()


# ============================================================
# 5. ORCHESTRATOR
# ============================================================
class Agent:
    def __init__(self):
        self.env = EmailEnv()
        self.model = WorldModel()
        self.memory = VectorMemory(dim=6)
        self.goal_system = GoalSystem()
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

    def rollout(self, state, first_action, horizon=3):
        current_state = np.asarray(state, dtype=np.float32).copy()
        total_score = 0.0
        discount = 1.0
        gamma = 0.85
        trajectory = []

        with torch.no_grad():
            for step in range(horizon):
                if step == 0:
                    action = first_action
                else:
                    action = max(
                        range(4),
                        key=lambda a: self.env.estimate_reward(current_state, a),
                    )

                state_t = torch.tensor(
                    current_state, dtype=torch.float32
                ).unsqueeze(0)
                action_t = torch.tensor([action], dtype=torch.int64)
                pred = self.model(state_t, action_t)
                next_state = np.asarray(
                    pred.squeeze(0).cpu().numpy(), dtype=np.float32
                )

                goal_value = self.goal_system.evaluate_state_value(next_state)

                similar = self.memory.retrieve(next_state, k=3)
                if similar:
                    memory_reward = float(
                        np.mean([float(m[3]) for m in similar])
                    )
                else:
                    memory_reward = 0.0

                immediate_reward = float(
                    self.env.estimate_reward(current_state, action)
                )
                risk_penalty = float(next_state[3]) * 0.3
                urgency_penalty = float(next_state[0]) * 0.2
                step_score = (
                    immediate_reward
                    + 0.4 * memory_reward
                    + 0.6 * goal_value
                    - risk_penalty
                    - urgency_penalty
                )

                total_score += discount * step_score
                discount *= gamma

                trajectory.append(
                    {
                        "action_name": self.env.ACTIONS[action],
                        "immediate_reward": immediate_reward,
                        "memory_reward": memory_reward,
                        "goal_value": goal_value,
                        "risk_penalty": risk_penalty,
                        "urgency_penalty": urgency_penalty,
                        "step_score": step_score,
                        "next_state": next_state,
                    }
                )
                current_state = next_state

        return {
            "first_action": first_action,
            "first_action_name": self.env.ACTIONS[first_action],
            "total_score": total_score,
            "trajectory": trajectory,
        }

    def rollout_all_actions(self, state, horizon=3):
        rollouts = [self.rollout(state, a, horizon) for a in range(4)]
        rollouts.sort(key=lambda r: r["total_score"], reverse=True)
        return rollouts

    def decide(self, state):
        rollouts = self.rollout_all_actions(state, horizon=3)
        print("Rollout simulation:")
        for r in sorted(rollouts, key=lambda x: x["first_action"]):
            print(
                f"- {r['first_action_name']} | total={r['total_score']:.2f}"
            )
            for si, step in enumerate(r["trajectory"]):
                print(
                    f"  step {si} -> action={step['action_name']} | "
                    f"immediate={step['immediate_reward']:.2f} | "
                    f"memory={step['memory_reward']:.2f} | "
                    f"goal={step['goal_value']:.2f} | "
                    f"score={step['step_score']:.2f}"
                )
        best = max(rollouts, key=lambda r: r["total_score"])
        final_predicted_state = best["trajectory"][-1]["next_state"]
        return (
            best["first_action"],
            best["total_score"],
            final_predicted_state,
            best,
        )

    def learn_from_outcome(self, final_state, total_score):
        self.goal_system.update_weights(final_state, total_score)


# ============================================================
# 6. MAIN
# ============================================================
if __name__ == "__main__":
    agent = Agent()
    for epoch in range(30):
        agent.collect()
        loss = agent.train()
        state, _ = agent.env.reset()
        print(f"Epoch {epoch}")
        print(f"State: {np.round(state, 2)}")
        action, total_score, final_predicted_state, _ = agent.decide(state)
        final_goal_value = agent.goal_system.evaluate_state_value(
            final_predicted_state
        )
        print(f"Decision: {agent.env.ACTIONS[action]}")
        print(f"Trajectory score: {total_score:.2f}")
        print(f"Final goal value: {final_goal_value:.2f}")
        print(f"Final predicted state: {np.round(final_predicted_state, 2)}")
        agent.learn_from_outcome(final_predicted_state, total_score)
        print("Adaptive weights:")
        for wkey in (
            "client_satisfaction",
            "risk_reduction",
            "urgency_control",
            "confidence_growth",
        ):
            print(f"- {wkey}: {agent.goal_system.weights[wkey]:.3f}")
        print(f"Loss: {loss}")
        print(f"Memory size: {len(agent.memory.storage)}")
