#!/usr/bin/env python
import os
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from vizdoom import DoomGame, scenarios_path
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cv2
import sys
import tqdm
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# VizDoom setup
def setup_vizdoom() -> DoomGame:
    game = DoomGame()
    game.load_config(os.path.join(scenarios_path, "basic.cfg"))
    game.set_ticrate(10)
    # don't render the screen
    game.set_window_visible(False)
    game.init()

    return game

# Define actions
ACTIONS = [
    [0, 0, 1],  # SHOOT
    [1, 0, 0],  # LEFT
    [0, 1, 0],  # RIGHT
]

# Hyperparameters
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPS_CLIP = 0.2
K_EPOCHS = 6
UPDATE_TIMESTEP = 32
EPISODES = 1000

class ActorCritic(nn.Module):
    def __init__(self, state_dim: Tuple[int, int, int], action_dim: int):
        super().__init__()

        self.shared_layers = nn.Sequential(
            nn.Conv2d(state_dim[0], 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4),
            nn.Flatten()
        )

        conv_out_size = 32*7*10

        self.actor = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
            nn.Softmax(dim=-1)
        )

        self.critic = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared_features = self.shared_layers[:-1](state)  # Get features before flattening
        flattened_features = self.shared_layers(state)
        action_probs = self.actor(flattened_features)
        state_value = self.critic(flattened_features)
        return action_probs, state_value, shared_features

    def act(self, state: torch.Tensor) -> Tuple[int, torch.Tensor, torch.Tensor]:
        action_probs, _, shared_features = self(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), shared_features

class PPO:
    def __init__(self, state_dim: Tuple[int, int, int], action_dim: int):
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LEARNING_RATE)
        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

    def update(self, states: List[np.ndarray], actions: List[int], logprobs: List[float],
               rewards: List[float], is_terminals: List[bool]) -> float:
        # Prepare data
        old_states = torch.FloatTensor(np.array(states)).to(device)
        old_actions = torch.LongTensor(actions).to(device)
        old_logprobs = torch.FloatTensor(logprobs).to(device)
        rewards = torch.FloatTensor(rewards).to(device)

        # Monte Carlo estimate of rewards
        returns = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rewards), reversed(is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (GAMMA * discounted_reward)
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Optimize policy for K epochs
        losses = 0
        for _ in range(K_EPOCHS):
            # Evaluating old actions and values
            action_probs, state_values, _ = self.policy(old_states)
            dist = Categorical(action_probs)
            logprobs = dist.log_prob(old_actions)
            dist_entropy = dist.entropy()
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = returns - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-EPS_CLIP, 1+EPS_CLIP) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*nn.MSELoss()(state_values, returns) - 0.01*dist_entropy

            # Take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            losses += loss.mean().item()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        return losses / K_EPOCHS

# Set up the plot for animation
fig, ax = plt.subplots()
# im = ax.imshow(np.zeros((7, 10)), cmap='viridis')
# plt.colorbar(im)
im = ax.imshow(np.zeros((120, 160*2, 3)))
# plt.colorbar(im)


def main():
    game = setup_vizdoom()
    state_dim = (3, 120, 160)  # Adjusted for the resized input
    action_dim = len(ACTIONS)

    ppo = PPO(state_dim, action_dim)
    timestep = 0
    episode = 0
    states, actions, logprobs, rewards, is_terminals, losses = [], [], [], [], [], []
    prgbar = tqdm.tqdm(desc="Training")
    def update_plot(_):
        try:
            nonlocal timestep, episode, states, actions, logprobs, rewards, is_terminals, losses
            timestep += 1
            state = game.get_state().screen_buffer / 255.0
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            state = F.interpolate(state, (120,160)).squeeze(0).cpu().numpy()
            with torch.no_grad():
                action, logprob, feature_maps = ppo.policy_old.act(torch.FloatTensor(state).unsqueeze(0).to(device))
            last_feature_map = feature_maps[0, :].cpu()#.numpy()


            reward = game.make_action(ACTIONS[action])
            done = game.is_episode_finished()


            states.append(state)
            actions.append(action)
            logprobs.append(logprob)
            rewards.append(reward)
            is_terminals.append(done)
            should_clean_up = False
            if (len(states) == UPDATE_TIMESTEP) or done:
                # print("Updating")
                prgbar.update(1)
                loss = ppo.update(states, actions, logprobs, rewards, is_terminals)
                # print(loss)
                losses.append(loss)
                should_clean_up = True

            prgbar.set_postfix(ep=episode, t=timestep)
            if done:

                _loss = sum(losses) / len(losses)
                prgbar.write(f"Episode {episode+1}, Total Reward: {sum(rewards)}, Loss: {_loss:.4f}")
                episode += 1
                game.new_episode()
                should_clean_up = True

            if should_clean_up:
                states, actions, logprobs, rewards, is_terminals, losses = [], [], [], [], [], []

            state = state.transpose(1,2,0)
            last_feature_map = last_feature_map.permute(1,2,0)
            last_feature_map = last_feature_map.chunk(3, dim=-1)
            last_feature_map = torch.stack([feat.mean(dim=-1) for feat in last_feature_map], dim=-1)
            last_feature_map = last_feature_map.numpy()
            last_feature_map = (last_feature_map - last_feature_map.min() )/ (last_feature_map.max() - last_feature_map.min())
            last_feature_map = cv2.resize(last_feature_map, (160, 120))


            image = np.concatenate((state, last_feature_map), axis=1)
            im.set_array(image)
            return [im]
        except Exception as ex:
            print(ex)
            sys.exit()

    ani = FuncAnimation(fig, update_plot, blit=True, interval=1)
    plt.show()
    game.close()

if __name__ == "__main__":
    main()
