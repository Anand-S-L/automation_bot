# Autonomous Game Navigator 🎮🤖

An intelligent bot framework for autonomous navigation in open-world games with real-time obstacle detection and avoidance. Specifically designed for Evil Lands running on **BlueStacks emulator**.

## ⚠️ Important Disclaimer

**This project is for EDUCATIONAL PURPOSES ONLY.** Using automation bots in online games:
- May violate the game's Terms of Service
- Can result in account bans
- May be considered cheating in multiplayer environments

Use this framework responsibly and only in single-player or test environments where automation is permitted.

## 🚀 Quick Start

### For BlueStacks Emulator Users (Recommended)

See **[BLUESTACKS_SETUP.md](BLUESTACKS_SETUP.md)** for complete setup guide!

**TL;DR:**
```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure minimap (first time only)
python configure_minimap.py

# 3. Run the bot
python minimap_navigator.py
```

### 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)** - ⚡ Quick reference card (START HERE!)
- **[BLUESTACKS_SETUP.md](BLUESTACKS_SETUP.md)** - Complete BlueStacks emulator setup guide
- **[MINIMAP_LOCATION.md](MINIMAP_LOCATION.md)** - 📍 Visual guide to find the minimap (top-right corner!)
- **[MINIMAP_GUIDE.md](MINIMAP_GUIDE.md)** - How minimap navigation works
- **[CONTROLS.md](CONTROLS.md)** - Control scheme (Arrow keys + Numpad)
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - 🔧 Common issues and solutions
- **[README.md](README.md)** - This file (general overview)

---

## 🌟 Features

- **Real-time Screen Capture**: Continuously monitors game screen
- **Computer Vision Obstacle Detection**: Uses multiple detection methods:
  - Darkness detection (walls, cliffs)
  - Edge detection (object boundaries)
  - Color-based detection (water, lava, etc.)
- **Autonomous Navigation**: Intelligently chooses safe paths
- **8-Directional Movement**: Smooth character control
- **Debug Visualization**: Real-time view of obstacle detection
- **Configurable**: Easy JSON configuration for different games
- **Camera Rotation**: Automatically adjusts view when stuck

## 🚀 How It Works

### Path Finding Logic

The bot doesn't use waypoints. Instead, it:

1. **Captures Screen**: Takes a screenshot of the game
2. **Detects Obstacles**: Analyzes the image to find:
   - Dark areas (walls, obstacles)
   - Color patterns (water, dangerous terrain)
   - Edge boundaries
3. **Analyzes Directions**: Divides view into 8 sectors and calculates "clearness score" for each
4. **Chooses Best Path**: Selects the direction with highest clearness
5. **Executes Movement**: Simulates keyboard input to move character
6. **Repeats**: Continuously adapts to new obstacles

### Obstacle Detection Methods

```
Screen → Grayscale → Darkness Detection
                   → Edge Detection      → Combined Mask → Direction Analysis
                   → Color Detection
```

## 📋 Prerequisites

- Python 3.8 or higher
- Windows, macOS, or Linux
- **BlueStacks emulator** (for Evil Lands)
- Game that can run in windowed mode (recommended)

## 🔧 Installation

### Step 1: Install Python Packages

```powershell
pip install -r requirements.txt
```

The required packages are:
- `opencv-python` - Computer vision and image processing
- `numpy` - Numerical operations
- `mss` - Fast screen capture
- `pyautogui` - Input simulation
- `pillow` - Image manipulation

### Step 2: Configure for Your Setup

**For BlueStacks:** See [BLUESTACKS_SETUP.md](BLUESTACKS_SETUP.md)

**For PC Games:** Continue with configuration below

---

## ⚙️ Configuration

### Step 1: Find Your Game Window Coordinates

Run the configuration helper:
```powershell
python configure_screen.py
```

This will:
1. Ask you to click the top-left corner of your game window
2. Ask you to click the bottom-right corner
3. Generate the correct `screen_region` values
4. Save a test capture image

### Step 2: Edit config.json

Update `config.json` with your settings:

```json
{
  "screen_region": [0, 0, 1920, 1080],
  "movement_duration": 0.3,
  "scan_interval": 0.5,
  "obstacle_threshold": 50,
  "turn_angle": 45,
  "exploration_bias": 0.7
}
```

**Configuration Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `screen_region` | `[left, top, width, height]` of game window | `[0, 0, 1920, 1080]` |
| `movement_duration` | How long to hold movement keys (seconds) | `0.3` |
| `scan_interval` | Time between screen scans (seconds) | `0.5` |
| `obstacle_threshold` | Darkness threshold (0-255, lower = more sensitive) | `50` |
| `turn_angle` | Camera rotation angle when stuck (degrees) | `45` |
| `exploration_bias` | Preference for unexplored areas (0-1) | `0.7` |

### Step 3: Tune for Your Game

Different games need different settings:

**Dark/Gothic Games** (Evil Lands, Dark Souls):
```json
{
  "obstacle_threshold": 40,
  "movement_duration": 0.4
}
```

**Bright/Colorful Games** (Zelda-style):
```json
{
  "obstacle_threshold": 70,
  "movement_duration": 0.3
}
```

**Fast-paced Games**:
```json
{
  "scan_interval": 0.3,
  "movement_duration": 0.2
}
```

## 🎮 Usage

### Basic Usage

1. **Start your game** and switch to windowed mode
2. **Position your character** in an open area
3. **Run the bot**:
```powershell
python autonomous_navigator.py
```
4. **Switch to game window** within 3 seconds
5. The bot will start navigating automatically

### Debug Mode

The bot runs in debug mode by default, showing:
- Left window: Game screen with clearness bars
- Right window: Obstacle detection mask (red = obstacles)
- Green bars: Chosen direction

To disable debug visualization, edit `autonomous_navigator.py`:
```python
navigator.start(debug=False)
```

### Stopping the Bot

Press `Ctrl+C` in the terminal to stop the bot safely.

## 🛠️ Customization for Evil Lands

For Evil Lands specifically:

1. **Screen Region**: Make sure to capture only the game view, excluding UI elements
2. **Obstacle Detection**: You may need to adjust for the game's color scheme:

Edit `ObstacleDetector.detect_obstacles()` method to add game-specific detection:

```python
# Detect Evil Lands specific obstacles
# Example: Detect red danger zones
red_lower = np.array([0, 100, 100])
red_upper = np.array([10, 255, 255])
danger_mask = cv2.inRange(hsv, red_lower, red_upper)
```

3. **Movement Keys**: If Evil Lands uses different controls, update `MovementController.DIRECTIONS`

## 📊 Understanding the Output

Console output:
```
✓ Moving FORWARD (clearness: 0.85)
✓ Moving FORWARD-RIGHT (clearness: 0.72)
⚠ Heavy obstacles detected - rotating...
✓ Moving FORWARD-LEFT (clearness: 0.68)
```

- `✓` = Successfully moving
- `⚠` = Detected obstacles, adjusting
- Clearness score: 0.0 = blocked, 1.0 = completely clear

## 🔍 Troubleshooting

### Bot doesn't move
- Check that game window is in focus
- Verify WASD keys control movement in your game
- Increase `movement_duration` in config

### Crashes into obstacles
- Lower `obstacle_threshold` (more sensitive)
- Reduce `scan_interval` (faster reactions)
- Adjust `screen_region` to avoid UI elements

### Stuttering movement
- Increase `scan_interval`
- Close debug window (`debug=False`)
- Reduce screen capture resolution

### Wrong game area captured
- Run `configure_screen.py` again
- Make sure game is in windowed mode
- Check `screen_region` values

## 🧩 Code Structure

```
autonomous_navigator.py
├── BotConfig           # Configuration management
├── ScreenCapture       # Game screen capture
├── ObstacleDetector    # Computer vision analysis
├── MovementController  # Input simulation
└── AutonomousNavigator # Main coordination logic
```

## 🎓 Learning Resources

This project demonstrates:
- **Computer Vision**: Image processing, edge detection, color analysis
- **Real-time Systems**: Continuous capture and processing
- **Decision Making**: Autonomous path selection
- **Input Automation**: Keyboard/mouse simulation
- **Game AI**: Obstacle avoidance, exploration strategies

## 📝 Future Enhancements

Possible improvements:
- [ ] Machine learning for better obstacle recognition
- [ ] Minimap integration for global navigation
- [ ] Path memory to avoid revisiting areas
- [ ] Enemy detection and avoidance
- [ ] Quest objective tracking
- [ ] A* pathfinding for specific targets

## 🤝 Contributing

This is an educational project. Feel free to:
- Experiment with different detection algorithms
- Add game-specific optimizations
- Improve pathfinding logic
- Share your configurations for different games

## 📜 License

This project is provided as-is for educational purposes. Use responsibly and in compliance with game Terms of Service.

---

**Remember**: Use this tool ethically and only where automation is permitted! 🛡️
