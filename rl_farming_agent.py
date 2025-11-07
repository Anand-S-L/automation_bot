"""
Reinforcement Learning Agent for Evil Lands Auto-Farming
Uses Deep Q-Network (DQN) to learn optimal farming behavior

This agent will:
1. Observe the game state (screen/minimap)
2. Learn which actions (movement, combat) lead to rewards (kills, loot, exp)
3. Improve over time through trial and error
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import time
from mss import mss
import pyautogui
from dataclasses import dataclass
from typing import Tuple, List, Optional
import json


@dataclass
class GameState:
    """Represents the current state of the game"""
    screen: np.ndarray  # Full game screen
    minimap: np.ndarray  # Minimap region
    health_percentage: float  # 0-100
    has_target: bool  # Is there an enemy targeted
    is_combat: bool  # Are we in combat
    timestamp: float  # Time of observation


class DQNetwork(nn.Module):
    """
    Deep Q-Network - Neural network that learns to predict action values
    
    Architecture:
    - Convolutional layers to process visual input
    - Fully connected layers for decision making
    """
    
    def __init__(self, input_channels=3, num_actions=15):  # Updated to 15 actions
        super(DQNetwork, self).__init__()
        
        # Convolutional layers to process screen/minimap
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate size after convolutions (depends on input size)
        # For 84x84 input: 7x7x64 = 3136
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, num_actions)
    
    def forward(self, x):
        """Forward pass through the network"""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Q-values for each action


class ReplayMemory:
    """
    Experience Replay Buffer
    Stores past experiences to learn from
    """
    
    def __init__(self, capacity=10000):
        self.memory = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Save an experience"""
        self.memory.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        """Sample a random batch of experiences"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)


class FarmingAgent:
    """
    The main RL agent that learns to farm
    """
    
    # Action space - EXPANDED with camera controls
    ACTIONS = {
        # Movement (8 directions)
        0: 'up',           # Move north
        1: 'down',         # Move south
        2: 'left',         # Move west
        3: 'right',        # Move east
        4: 'up+left',      # Move northwest
        5: 'up+right',     # Move northeast
        6: 'down+left',    # Move southwest
        7: 'down+right',   # Move southeast
        
        # Combat
        8: 'attack',       # Attack nearest enemy
        
        # Camera control (numpad keys for Evil Lands)
        9: 'look_left',    # Numpad 4 - Look left
        10: 'look_right',  # Numpad 6 - Look right
        11: 'look_up',     # Numpad 8 - Look up
        12: 'look_down',   # Numpad 5 - Look down
        
        # Combined actions for efficiency
        13: 'attack+look_left',   # Attack while adjusting camera left
        14: 'attack+look_right',  # Attack while adjusting camera right
    }
    
    def __init__(self, config_path='config_rl.json'):
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Setup device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Create neural networks
        self.policy_net = DQNetwork().to(self.device)
        self.target_net = DQNetwork().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        
        # Replay memory
        self.memory = ReplayMemory(capacity=self.config['memory_size'])
        
        # Training parameters
        self.batch_size = 32
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate (starts high)
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        # Game state tracking
        self.last_state = None
        self.last_action = None
        self.episode_reward = 0
        self.steps = 0
        
        # Screen capture
        self.sct = mss()
        self.game_region = self.config['game_region']
        self.minimap_region = self.config['minimap_region']
        
        # Statistics
        self.episode_rewards = []
        self.kills = 0
        self.deaths = 0
    
    def _load_config(self, path):
        """Load configuration from JSON"""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # Default configuration
            return {
                'game_region': [0, 0, 1920, 1080],
                'minimap_region': [1670, 50, 200, 200],
                'memory_size': 10000,
                'save_interval': 100,  # Save model every N episodes
                'target_update': 10,   # Update target network every N episodes
            }
    
    def capture_game_state(self) -> GameState:
        """Capture current game state from screen"""
        # Capture game screen
        screenshot = self.sct.grab(self.game_region)
        screen = np.array(screenshot)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
        
        # Capture minimap
        minimap_shot = self.sct.grab(self.minimap_region)
        minimap = np.array(minimap_shot)
        minimap = cv2.cvtColor(minimap, cv2.COLOR_BGRA2BGR)
        
        # TODO: Extract health, target status, combat status from screen
        # For now, using placeholder values
        health = 100.0
        has_target = False
        is_combat = False
        
        return GameState(
            screen=screen,
            minimap=minimap,
            health_percentage=health,
            has_target=has_target,
            is_combat=is_combat,
            timestamp=time.time()
        )
    
    def preprocess_state(self, state: GameState) -> torch.Tensor:
        """
        Preprocess game state for neural network input
        - Resize to 84x84
        - Convert to grayscale or keep color
        - Normalize pixel values
        """
        # Use minimap for now (smaller, easier to process)
        img = state.minimap
        
        # Resize to 84x84
        img = cv2.resize(img, (84, 84))
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        # Transpose to (C, H, W) format for PyTorch
        img = np.transpose(img, (2, 0, 1))
        
        # Convert to tensor
        tensor = torch.from_numpy(img).unsqueeze(0).to(self.device)
        
        return tensor
    
    def select_action(self, state_tensor: torch.Tensor, training=True) -> int:
        """
        Select action using epsilon-greedy policy
        - With probability epsilon: random action (exploration)
        - Otherwise: best action according to network (exploitation)
        """
        if training and random.random() < self.epsilon:
            # Exploration: random action
            return random.randrange(len(self.ACTIONS))
        else:
            # Exploitation: best action from network
            with torch.no_grad():
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
    
    def execute_action(self, action_id: int):
        """Execute the selected action in the game"""
        action = self.ACTIONS[action_id]
        
        # Parse action components
        action_parts = action.split('+')
        
        for part in action_parts:
            if part == 'attack':
                # Press attack key (space bar for Evil Lands)
                pyautogui.press('space')
                
            elif part == 'look_left':
                # Numpad 4 - Look left
                pyautogui.keyDown('num4')
                time.sleep(0.15)
                pyautogui.keyUp('num4')
                
            elif part == 'look_right':
                # Numpad 6 - Look right
                pyautogui.keyDown('num6')
                time.sleep(0.15)
                pyautogui.keyUp('num6')
                
            elif part == 'look_up':
                # Numpad 8 - Look up
                pyautogui.keyDown('num8')
                time.sleep(0.15)
                pyautogui.keyUp('num8')
                
            elif part == 'look_down':
                # Numpad 5 - Look down
                pyautogui.keyDown('num5')
                time.sleep(0.15)
                pyautogui.keyUp('num5')
                
            elif part in ['up', 'down', 'left', 'right']:
                # Movement keys
                # Release all movement keys first
                for key in ['up', 'down', 'left', 'right']:
                    pyautogui.keyUp(key)
                
                # Press the movement key
                pyautogui.keyDown(part)
                time.sleep(0.1)
    
    def calculate_reward(self, prev_state: GameState, curr_state: GameState, action_id: int) -> float:
        """
        Calculate reward for the transition
        
        Reward structure:
        - Positive reward for: kills, collecting loot, gaining exp, staying alive
        - Negative reward for: taking damage, dying, getting stuck
        - Small negative reward each step (encourages efficiency)
        """
        reward = -0.01  # Small step penalty (encourages efficiency)
        
        # TODO: Implement proper reward detection from screen
        # This requires OCR or template matching to detect:
        # - Damage numbers
        # - Kill notifications
        # - Loot pickups
        # - Health changes
        
        # Placeholder reward logic:
        # If health decreased, negative reward
        if curr_state.health_percentage < prev_state.health_percentage:
            reward -= (prev_state.health_percentage - curr_state.health_percentage) * 0.1
        
        # If in combat, small positive reward (we want to fight)
        if curr_state.is_combat:
            reward += 0.05
        
        # TODO: Add rewards for:
        # - Kills: +10
        # - Loot: +5
        # - Exp gain: +1
        # - Death: -50
        
        return reward
    
    def train_step(self):
        """Perform one training step (update network weights)"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        batch = self.memory.sample(self.batch_size)
        
        # Unpack batch
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        state_batch = torch.cat(states)
        action_batch = torch.tensor(actions, device=self.device)
        reward_batch = torch.tensor(rewards, device=self.device, dtype=torch.float32)
        next_state_batch = torch.cat(next_states)
        done_batch = torch.tensor(dones, device=self.device, dtype=torch.float32)
        
        # Current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Next Q values from target network
        next_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        
        # Expected Q values
        expected_q_values = reward_batch + (self.gamma * next_q_values * (1 - done_batch))
        
        # Compute loss
        loss = F.smooth_l1_loss(current_q_values.squeeze(), expected_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
    
    def train(self, num_episodes=1000):
        """
        Main training loop
        """
        print("=" * 60)
        print("REINFORCEMENT LEARNING FARMING AGENT")
        print("=" * 60)
        print(f"\nTraining for {num_episodes} episodes...")
        print(f"Device: {self.device}")
        print(f"Initial epsilon: {self.epsilon}")
        print("\nStarting in 3 seconds...")
        time.sleep(3)
        
        for episode in range(num_episodes):
            print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
            
            # Reset episode
            self.episode_reward = 0
            done = False
            step = 0
            max_steps = 1000  # Max steps per episode
            
            # Get initial state
            state = self.capture_game_state()
            state_tensor = self.preprocess_state(state)
            
            while not done and step < max_steps:
                # Select and execute action
                action = self.select_action(state_tensor, training=True)
                self.execute_action(action)
                
                # Wait for game to respond
                time.sleep(0.2)
                
                # Observe new state
                next_state = self.capture_game_state()
                next_state_tensor = self.preprocess_state(next_state)
                
                # Calculate reward
                reward = self.calculate_reward(state, next_state, action)
                self.episode_reward += reward
                
                # Check if episode is done (died, reached goal, etc.)
                done = next_state.health_percentage <= 0
                
                # Store experience in memory
                self.memory.push(state_tensor, action, reward, next_state_tensor, done)
                
                # Train on batch
                self.train_step()
                
                # Move to next state
                state = next_state
                state_tensor = next_state_tensor
                step += 1
                
                # Print progress
                if step % 50 == 0:
                    print(f"  Step {step} | Reward: {self.episode_reward:.2f} | Epsilon: {self.epsilon:.3f}")
            
            # Episode finished
            self.episode_rewards.append(self.episode_reward)
            
            # Decay epsilon (explore less over time)
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Update target network periodically
            if (episode + 1) % self.config['target_update'] == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())
                print(f"  → Target network updated")
            
            # Save model periodically
            if (episode + 1) % self.config['save_interval'] == 0:
                self.save_model(f"model_episode_{episode + 1}.pth")
                print(f"  → Model saved")
            
            # Print episode summary
            avg_reward = np.mean(self.episode_rewards[-10:]) if len(self.episode_rewards) >= 10 else self.episode_reward
            print(f"Episode reward: {self.episode_reward:.2f} | Avg (last 10): {avg_reward:.2f}")
    
    def save_model(self, filename='farming_agent.pth'):
        """Save the trained model"""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
        }, filename)
        print(f"Model saved to {filename}")
    
    def load_model(self, filename='farming_agent.pth'):
        """Load a trained model"""
        checkpoint = torch.load(filename)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        print(f"Model loaded from {filename}")
    
    def play(self, num_episodes=10):
        """Play using the trained agent (no exploration)"""
        self.epsilon = 0  # No exploration
        
        for episode in range(num_episodes):
            print(f"\n=== Playing Episode {episode + 1}/{num_episodes} ===")
            
            state = self.capture_game_state()
            state_tensor = self.preprocess_state(state)
            
            done = False
            step = 0
            episode_reward = 0
            
            while not done and step < 1000:
                action = self.select_action(state_tensor, training=False)
                self.execute_action(action)
                time.sleep(0.2)
                
                next_state = self.capture_game_state()
                next_state_tensor = self.preprocess_state(next_state)
                
                reward = self.calculate_reward(state, next_state, action)
                episode_reward += reward
                
                done = next_state.health_percentage <= 0
                
                state = next_state
                state_tensor = next_state_tensor
                step += 1
            
            print(f"Episode reward: {episode_reward:.2f}")


def main():
    """Main entry point"""
    print("Evil Lands Reinforcement Learning Farming Agent")
    print("=" * 60)
    print("\nThis agent will learn to farm automatically using Deep Q-Learning")
    print("\nOptions:")
    print("  1. Train new agent")
    print("  2. Continue training existing agent")
    print("  3. Play with trained agent")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    agent = FarmingAgent()
    
    if choice == '1':
        # Train from scratch
        episodes = int(input("Number of episodes to train (default 1000): ") or "1000")
        agent.train(num_episodes=episodes)
        agent.save_model()
    
    elif choice == '2':
        # Continue training
        model_file = input("Model file to load (default 'farming_agent.pth'): ") or "farming_agent.pth"
        agent.load_model(model_file)
        episodes = int(input("Additional episodes to train (default 500): ") or "500")
        agent.train(num_episodes=episodes)
        agent.save_model()
    
    elif choice == '3':
        # Play with trained model
        model_file = input("Model file to load (default 'farming_agent.pth'): ") or "farming_agent.pth"
        agent.load_model(model_file)
        episodes = int(input("Number of episodes to play (default 10): ") or "10")
        agent.play(num_episodes=episodes)
    
    else:
        print("Invalid choice!")


if __name__ == "__main__":
    main()
