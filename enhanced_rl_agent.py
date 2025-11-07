"""
Updated RL Farming Agent with Full Perception System (Phase 2 Complete)
This file integrates all perception modules into the RL agent
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import cv2
import mss
import pyautogui
import time
from dataclasses import dataclass
from collections import deque
from typing import Tuple, List, Optional, Dict
import json

# Import perception modules
from perception.health_detection import HealthDetector, HealthManaState
from perception.enemy_detection import EnemyDetector, EnemyState
from perception.reward_detection import RewardDetector, RewardEvent, RewardState


@dataclass
class EnhancedGameState:
    """
    Enhanced game state with full perception
    """
    # Visual input
    screen: np.ndarray
    minimap: np.ndarray
    
    # Health/Mana (from HealthDetector)
    health_percentage: float
    mana_percentage: float
    is_low_health: bool
    is_low_mana: bool
    is_critical: bool
    
    # Enemies (from EnemyDetector)
    enemy_count: int
    has_target: bool
    nearest_enemy_distance: float  # 0-1
    enemies_visible: List  # Full enemy list
    
    # Rewards (from RewardDetector)
    recent_kills: int  # In last 5 seconds
    recent_loot: int
    recent_damage_dealt: float
    recent_damage_taken: float
    
    # Combat state
    is_in_combat: bool
    
    # Timestamp
    timestamp: float
    
    def to_state_vector(self) -> np.ndarray:
        """
        Convert to fixed-size state vector for neural network
        
        Returns:
            State vector [health, mana, enemy_count, has_target, distance, ...]
        """
        return np.array([
            self.health_percentage / 100.0,
            self.mana_percentage / 100.0,
            float(self.is_low_health),
            float(self.is_low_mana),
            float(self.is_critical),
            min(self.enemy_count / 10.0, 1.0),  # Normalize to 0-1
            float(self.has_target),
            self.nearest_enemy_distance,
            float(self.is_in_combat),
            min(self.recent_kills / 5.0, 1.0),
            min(self.recent_loot / 5.0, 1.0),
        ], dtype=np.float32)


class EnhancedDQNetwork(nn.Module):
    """
    Enhanced DQN with dual-input architecture
    
    Input 1: Visual (screen + minimap) → CNN
    Input 2: Game state vector → FC layers
    Combined → Q-values
    """
    
    def __init__(self, num_actions=15):
        super(EnhancedDQNetwork, self).__init__()
        
        # Visual stream (CNN)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate conv output size (for 84x84 input)
        self.conv_output_size = 3136
        
        # State vector stream (FC layers)
        self.state_fc1 = nn.Linear(11, 128)  # 11 state features
        self.state_fc2 = nn.Linear(128, 256)
        
        # Combined stream
        self.fc1 = nn.Linear(self.conv_output_size + 256, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
        self.relu = nn.ReLU()
        
    def forward(self, visual_input, state_vector):
        """
        Forward pass
        
        Args:
            visual_input: (batch, 3, 84, 84) screen image
            state_vector: (batch, 11) game state features
        """
        # Visual stream
        x_vis = self.relu(self.conv1(visual_input))
        x_vis = self.relu(self.conv2(x_vis))
        x_vis = self.relu(self.conv3(x_vis))
        x_vis = x_vis.view(x_vis.size(0), -1)
        
        # State stream
        x_state = self.relu(self.state_fc1(state_vector))
        x_state = self.relu(self.state_fc2(x_state))
        
        # Combine
        x = torch.cat([x_vis, x_state], dim=1)
        x = self.relu(self.fc1(x))
        q_values = self.fc2(x)
        
        return q_values


class EnhancedFarmingAgent:
    """
    State-of-the-art RL farming agent with full perception
    """
    
    # Action space (same as before)
    ACTIONS = {
        0: {'name': 'forward', 'keys': ['w']},
        1: {'name': 'backward', 'keys': ['s']},
        2: {'name': 'left', 'keys': ['a']},
        3: {'name': 'right', 'keys': ['d']},
        4: {'name': 'forward_left', 'keys': ['w', 'a']},
        5: {'name': 'forward_right', 'keys': ['w', 'd']},
        6: {'name': 'backward_left', 'keys': ['s', 'a']},
        7: {'name': 'backward_right', 'keys': ['s', 'd']},
        8: {'name': 'attack', 'keys': ['space']},
        9: {'name': 'look_left', 'keys': ['num4']},
        10: {'name': 'look_right', 'keys': ['num6']},
        11: {'name': 'look_up', 'keys': ['num8']},
        12: {'name': 'look_down', 'keys': ['num5']},
        13: {'name': 'attack_look_left', 'keys': ['space', 'num4']},
        14: {'name': 'attack_look_right', 'keys': ['space', 'num6']},
    }
    
    def __init__(self, config_path: Optional[str] = "config_rl.json"):
        """Initialize enhanced agent"""
        print("[EnhancedFarmingAgent] Initializing state-of-the-art RL agent...")
        
        # Load config
        self.config = self._load_config(config_path)
        
        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  Using device: {self.device}")
        
        # Initialize perception modules
        print("  Initializing perception systems...")
        self.health_detector = HealthDetector(self.config.get('health_detection'))
        self.enemy_detector = EnemyDetector(self.config.get('enemy_detection'))
        self.reward_detector = RewardDetector(self.config.get('reward_detection'))
        
        # Initialize neural networks
        print("  Building neural networks...")
        self.policy_net = EnhancedDQNetwork(num_actions=len(self.ACTIONS)).to(self.device)
        self.target_net = EnhancedDQNetwork(num_actions=len(self.ACTIONS)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.config.get('learning_rate', 1e-4))
        
        # Experience replay
        self.memory = deque(maxlen=self.config.get('memory_size', 10000))
        
        # Training parameters
        self.gamma = self.config.get('gamma', 0.99)
        self.epsilon = self.config.get('epsilon_start', 1.0)
        self.epsilon_end = self.config.get('epsilon_end', 0.01)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.batch_size = self.config.get('batch_size', 32)
        
        # Screen capture
        self.sct = mss.mss()
        self.game_region = self.config.get('game_region', [0, 0, 1920, 1080])
        self.minimap_region = self.config.get('minimap_region', [1670, 50, 200, 200])
        
        # State tracking
        self.current_state = None
        self.prev_screen = None
        self.episode_rewards = []
        self.episode_steps = 0
        
        # TensorBoard (if available)
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(f'runs/farming_agent_{int(time.time())}')
            self.use_tensorboard = True
        except ImportError:
            print("  TensorBoard not available (install: pip install tensorboard)")
            self.use_tensorboard = False
        
        print("[EnhancedFarmingAgent] Initialization complete!\n")
        print(f"  Actions: {len(self.ACTIONS)}")
        print(f"  Memory size: {len(self.memory)}")
        print(f"  Epsilon: {self.epsilon}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"  Config not found, using defaults")
            return {
                'game_region': [0, 0, 1920, 1080],
                'minimap_region': [1670, 50, 200, 200],
                'learning_rate': 1e-4,
                'memory_size': 10000,
                'gamma': 0.99,
                'epsilon_start': 1.0,
                'epsilon_end': 0.01,
                'epsilon_decay': 0.995,
                'batch_size': 32,
            }
    
    def capture_enhanced_state(self) -> EnhancedGameState:
        """
        Capture full game state with all perception systems
        """
        # Capture screen
        game_region_dict = {
            "left": self.game_region[0],
            "top": self.game_region[1],
            "width": self.game_region[2],
            "height": self.game_region[3]
        }
        screenshot = self.sct.grab(game_region_dict)
        screen = np.array(screenshot)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
        
        # Extract minimap
        mx, my, mw, mh = self.minimap_region
        minimap = screen[my:my+mh, mx:mx+mw]
        
        # Run perception systems
        health_state = self.health_detector.detect(screen)
        enemy_state = self.enemy_detector.detect(screen, minimap)
        reward_events = self.reward_detector.detect(screen, self.prev_screen)
        
        # Count recent events (last 5 seconds)
        current_time = time.time()
        recent_kills = sum(1 for e in self.reward_detector.event_history 
                          if e.event_type == 'kill' and current_time - e.timestamp < 5.0)
        recent_loot = sum(1 for e in self.reward_detector.event_history 
                         if e.event_type == 'loot' and current_time - e.timestamp < 5.0)
        recent_damage_dealt = sum(e.value for e in self.reward_detector.event_history 
                                  if e.event_type == 'damage_dealt' and current_time - e.timestamp < 2.0)
        recent_damage_taken = sum(e.value for e in self.reward_detector.event_history 
                                  if e.event_type == 'damage_taken' and current_time - e.timestamp < 2.0)
        
        # Determine combat state (has enemies or taking damage)
        is_in_combat = enemy_state.enemy_count > 0 or recent_damage_taken > 0
        
        # Build state
        state = EnhancedGameState(
            screen=screen,
            minimap=minimap,
            health_percentage=health_state.health_percentage,
            mana_percentage=health_state.mana_percentage,
            is_low_health=health_state.is_low_health,
            is_low_mana=health_state.is_low_mana,
            is_critical=health_state.is_critical,
            enemy_count=enemy_state.enemy_count,
            has_target=enemy_state.has_target,
            nearest_enemy_distance=enemy_state.nearest_enemy.distance_score if enemy_state.nearest_enemy else 0.0,
            enemies_visible=enemy_state.enemies,
            recent_kills=recent_kills,
            recent_loot=recent_loot,
            recent_damage_dealt=recent_damage_dealt,
            recent_damage_taken=recent_damage_taken,
            is_in_combat=is_in_combat,
            timestamp=current_time
        )
        
        # Save for next frame
        self.prev_screen = screen.copy()
        
        return state
    
    def calculate_enhanced_reward(self, state: EnhancedGameState, prev_state: Optional[EnhancedGameState], 
                                   reward_events: List[RewardEvent]) -> float:
        """
        Calculate reward with full perception data
        
        Reward structure:
        - Kill: +10.0
        - Loot (common): +5.0, (rare): +10.0, (epic): +20.0
        - XP gain: +0.1 per point
        - Damage dealt: +0.1 per point
        - Damage taken: -0.5 per point
        - Death: -50.0
        - Moving toward enemy: +0.01
        - Low health survival: +0.05 per step
        - Idle (no action): -0.5
        """
        reward = 0.0
        
        # Immediate rewards from events
        for event in reward_events:
            if event.event_type == 'kill':
                reward += 10.0 * event.confidence
            elif event.event_type == 'loot':
                reward += event.value * 5.0  # Value = rarity (1-5)
            elif event.event_type == 'xp':
                reward += event.value * 0.1
            elif event.event_type == 'death':
                reward -= 50.0
            elif event.event_type == 'level_up':
                reward += 25.0
            elif event.event_type == 'damage_dealt':
                reward += 0.1
            elif event.event_type == 'damage_taken':
                reward -= 0.5
        
        # Shaped rewards (guide learning)
        if prev_state:
            # Reward for moving toward enemies
            if state.nearest_enemy_distance > prev_state.nearest_enemy_distance:
                reward += 0.01
            
            # Reward for staying alive with low health
            if state.is_low_health and not state.is_critical:
                reward += 0.05
            
            # Penalty for health loss
            health_loss = prev_state.health_percentage - state.health_percentage
            if health_loss > 0:
                reward -= health_loss * 0.1
        
        # Combat engagement bonus
        if state.is_in_combat and state.has_target:
            reward += 0.02
        
        # Small step penalty to encourage efficiency
        reward -= 0.01
        
        return reward
    
    def select_action(self, state: EnhancedGameState) -> int:
        """Select action using epsilon-greedy policy"""
        if np.random.random() < self.epsilon:
            return np.random.randint(0, len(self.ACTIONS))
        
        with torch.no_grad():
            # Preprocess visual
            visual = self._preprocess_visual(state.screen)
            visual = visual.unsqueeze(0).to(self.device)
            
            # Get state vector
            state_vector = torch.FloatTensor(state.to_state_vector()).unsqueeze(0).to(self.device)
            
            # Get Q-values
            q_values = self.policy_net(visual, state_vector)
            return q_values.argmax().item()
    
    def _preprocess_visual(self, screen: np.ndarray) -> torch.Tensor:
        """Preprocess screen for neural network"""
        # Resize
        resized = cv2.resize(screen, (84, 84))
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to (C, H, W)
        transposed = np.transpose(normalized, (2, 0, 1))
        
        return torch.FloatTensor(transposed)
    
    def execute_action(self, action_id: int):
        """Execute action in game"""
        action = self.ACTIONS[action_id]
        
        for key in action['keys']:
            pyautogui.keyDown(key)
        
        time.sleep(0.05)  # Hold briefly
        
        for key in action['keys']:
            pyautogui.keyUp(key)
        
        time.sleep(0.05)
    
    def train_step(self):
        """Perform one training step"""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch
        batch = np.random.choice(len(self.memory), self.batch_size, replace=False)
        
        visual_batch = []
        state_batch = []
        action_batch = []
        reward_batch = []
        next_visual_batch = []
        next_state_batch = []
        done_batch = []
        
        for idx in batch:
            s, a, r, s_next, done = self.memory[idx]
            visual_batch.append(self._preprocess_visual(s.screen))
            state_batch.append(s.to_state_vector())
            action_batch.append(a)
            reward_batch.append(r)
            next_visual_batch.append(self._preprocess_visual(s_next.screen))
            next_state_batch.append(s_next.to_state_vector())
            done_batch.append(done)
        
        # Convert to tensors
        visual_batch = torch.stack(visual_batch).to(self.device)
        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.LongTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        next_visual_batch = torch.stack(next_visual_batch).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        done_batch = torch.FloatTensor(done_batch).to(self.device)
        
        # Compute Q(s, a)
        q_values = self.policy_net(visual_batch, state_batch)
        q_values = q_values.gather(1, action_batch.unsqueeze(1)).squeeze()
        
        # Compute target
        with torch.no_grad():
            next_q_values = self.target_net(next_visual_batch, next_state_batch).max(1)[0]
            target = reward_batch + (1 - done_batch) * self.gamma * next_q_values
        
        # Loss
        loss = nn.functional.smooth_l1_loss(q_values, target)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def save_model(self, path: str = "enhanced_farming_agent.pth"):
        """Save model checkpoint"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode_rewards': self.episode_rewards,
        }, path)
        print(f"[EnhancedFarmingAgent] Model saved to {path}")
    
    def load_model(self, path: str = "enhanced_farming_agent.pth"):
        """Load model checkpoint"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        print(f"[EnhancedFarmingAgent] Model loaded from {path}")


# Example usage
if __name__ == "__main__":
    print("Enhanced RL Farming Agent - Phase 2 Complete")
    print("=" * 60)
    print("\nThis is the integrated agent with full perception.")
    print("See rl_farming_agent.py for training/playing functions.")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements_rl.txt")
    print("2. Test perception: python perception/health_detection.py")
    print("3. Calibrate detectors for your game")
    print("4. Run training with enhanced agent")
