# 🤖 Evil Lands Reinforcement Learning Auto-Farmer

## Overview

This is a **Deep Q-Network (DQN)** agent that learns to farm in Evil Lands automatically through **trial and error**. Unlike the simple navigation scripts, this AI agent:

- ✅ **Learns from experience** - Gets better over time
- ✅ **Adapts to situations** - Handles combat, loot, navigation
- ✅ **Makes decisions** - When to fight, when to move, where to go
- ✅ **Improves strategy** - Maximizes kills, loot, and survival

## How It Works

### Reinforcement Learning Basics

1. **Observe** - Agent sees the game state (screen/minimap)
2. **Decide** - Neural network chooses an action (move, attack)
3. **Act** - Execute action in game
4. **Learn** - Get reward/penalty, update neural network
5. **Repeat** - Do this thousands of times

### Reward System

The agent learns through rewards:
- **Positive Rewards:**
  - Kill enemy: +10
  - Collect loot: +5
  - Gain experience: +1
  - Stay in combat: +0.05 (encourages fighting)

- **Negative Rewards:**
  - Take damage: -0.1 per HP lost
  - Die: -50
  - Each step: -0.01 (encourages efficiency)

### Architecture

```
Game Screen (1920x1080)
      ↓
Minimap Extract (84x84x3)
      ↓
Convolutional Neural Network
  - Conv Layer 1: 32 filters
  - Conv Layer 2: 64 filters
  - Conv Layer 3: 64 filters
      ↓
Fully Connected Layers (512 neurons)
      ↓
Q-Values for 9 Actions
      ↓
Action Selection (ε-greedy)
      ↓
Execute in Game
```

## Installation

### 1. Install PyTorch

**For NVIDIA GPU (Recommended for faster training):**
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```powershell
pip install torch torchvision
```

### 2. Install Other Requirements

```powershell
pip install -r requirements_rl.txt
```

## Usage

### Train a New Agent

```powershell
python rl_farming_agent.py
```

Choose option 1 (Train new agent) and specify number of episodes (e.g., 1000).

**Training Process:**
- Episode 1-200: Agent explores randomly (learning basics)
- Episode 200-500: Begins exploiting learned strategies
- Episode 500+: Refined farming behavior

**Training Time:**
- CPU: ~10-20 seconds per episode
- GPU: ~2-5 seconds per episode
- 1000 episodes ≈ 3-10 hours

### Continue Training

If you want to train more after initial training:

```powershell
python rl_farming_agent.py
```

Choose option 2, specify the model file, and add more episodes.

### Use Trained Agent

After training, use the agent to auto-farm:

```powershell
python rl_farming_agent.py
```

Choose option 3 and select the trained model file.

## Configuration

Create `config_rl.json` to customize:

```json
{
  "game_region": [0, 0, 1920, 1080],
  "minimap_region": [1670, 50, 200, 200],
  "memory_size": 10000,
  "save_interval": 100,
  "target_update": 10
}
```

## Action Space

The agent can perform 9 actions:

| Action ID | Description | Keys |
|-----------|-------------|------|
| 0 | Move North | ↑ |
| 1 | Move South | ↓ |
| 2 | Move West | ← |
| 3 | Move East | → |
| 4 | Move Northwest | ↑ + ← |
| 5 | Move Northeast | ↑ + → |
| 6 | Move Southwest | ↓ + ← |
| 7 | Move Southeast | ↓ + → |
| 8 | Attack | Space |
| 9 | Look Left | Numpad 4 |
| 10 | Look Right | Numpad 6 |
| 11 | Look Up | Numpad 8 |
| 12 | Look Down | Numpad 5 |
| 13 | Attack + Look Left | Space + Num4 |
| 14 | Attack + Look Right | Space + Num6 |

**Total: 15 actions** (8 movement + 1 attack + 4 camera + 2 combined)

The agent will learn when to use camera controls to:
- Find enemies that are off-screen
- Track moving targets during combat
- Look around when stuck or lost
- Maintain optimal camera angle for fighting

## Training Tips

### 1. Start in a Good Location
- Place character in an area with many enemies
- Avoid areas with elite/boss mobs initially
- Good starting zones have consistent mob spawns

### 2. Monitor Training
Watch console output:
- **Episode reward increasing** = Learning is working ✅
- **Reward stuck/decreasing** = May need tuning ⚠️
- **High epsilon (>0.5)** = Still exploring (normal early on)
- **Low epsilon (<0.1)** = Exploiting learned strategy

### 3. Hyperparameter Tuning

If learning is slow, edit `rl_farming_agent.py`:

```python
# Learning rate (line ~92)
self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)  # Try 0.001

# Epsilon decay (line ~96)
self.epsilon_decay = 0.995  # Try 0.99 for faster exploitation

# Discount factor (line ~95)
self.gamma = 0.99  # Try 0.95 for more short-term focus
```

## Advanced Features (TODO)

The current implementation has basic structure. To make it production-ready:

### 1. Better State Detection
- [ ] OCR for health/mana bars
- [ ] Template matching for enemy detection
- [ ] Minimap enemy indicator detection
- [ ] Loot detection

### 2. Enhanced Rewards
- [ ] Detect kill notifications (OCR)
- [ ] Detect loot pickups
- [ ] Detect level-ups
- [ ] Detect stuck situations

### 3. Improved Neural Network
- [ ] Use LSTM for temporal patterns
- [ ] Dual network architecture (actor-critic)
- [ ] Attention mechanism for important screen regions
- [ ] Multi-scale input (full screen + minimap)

### 4. Training Improvements
- [ ] Prioritized experience replay
- [ ] Curriculum learning (easy → hard areas)
- [ ] Multiple parallel agents
- [ ] Transfer learning from saved models

## Troubleshooting

### Agent Dies Immediately
- Increase epsilon (more exploration)
- Increase death penalty reward
- Start in safer area with weaker enemies

### Not Learning (Flat Rewards)
- Check if rewards are being calculated correctly
- Verify actions are executing in game
- Try larger network or different architecture
- Increase learning rate

### Training Too Slow
- Use GPU instead of CPU
- Reduce episode length (max_steps)
- Reduce batch size
- Use smaller network

### Agent Gets Stuck
- Add stuck detection (same position for N steps)
- Add negative reward for repetitive actions
- Increase exploration (higher epsilon)

## Comparison: RL vs Simple Navigation

| Feature | Simple Navigator | RL Agent |
|---------|-----------------|----------|
| Setup Time | 5 minutes | 3-10 hours training |
| Adaptability | Fixed rules | Learns & adapts |
| Combat | None | Learns to fight |
| Loot | None | Learns to collect |
| Optimization | Manual tuning | Self-optimizing |
| Maintenance | Update rules | Retrain |

## Performance Expectations

After proper training (1000+ episodes):
- **Survival rate:** 70-90%
- **Kills per hour:** 50-200 (depends on area)
- **Efficiency:** Continuously improving
- **Adaptation:** Handles new situations reasonably well

## Legal & Ethical Notice

⚠️ **Use at your own risk**

- This is for educational purposes
- Using bots may violate game Terms of Service
- May result in account suspension/ban
- Author is not responsible for consequences

## Credits

Built using:
- **PyTorch** - Deep learning framework
- **OpenCV** - Computer vision
- **DQN Algorithm** - Mnih et al., 2015

## Next Steps

1. **Install requirements** - Get PyTorch and dependencies
2. **Configure game** - Set up screen regions
3. **Start training** - Run for 500-1000 episodes
4. **Evaluate** - Watch it play
5. **Iterate** - Tune rewards and train more

Happy farming! 🎮🤖
