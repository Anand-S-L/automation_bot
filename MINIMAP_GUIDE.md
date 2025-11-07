# Minimap-Based Navigation for Evil Lands 🗺️

## Why Minimap Navigation is BETTER

Using the minimap instead of 3D vision has major advantages:

✅ **Much more reliable** - Clear 2D view vs complex 3D scene  
✅ **Follows narrow paths** - Can detect and follow trails  
✅ **Better obstacle detection** - Minimap shows exact walkable areas  
✅ **Less processing** - Analyzing small minimap vs full screen  
✅ **Works in all conditions** - Night, fog, visual effects don't affect minimap  

## How It Works

Looking at your Evil Lands screenshot, the minimap shows:

1. **Light/White areas** = Walkable paths
2. **Dark/Black areas** = Obstacles, walls, cliffs
3. **Green dot** = Your character
4. **Narrow lines** = Trails and pathways

**Location:** The minimap in Evil Lands is the **largest circle on the top-right corner**, positioned **slightly down** from the very top edge of the screen.

The bot:
1. Captures just the minimap region (top-left circle in your game)
2. Detects light paths vs dark obstacles
3. Uses **path skeleton algorithm** to follow narrow trails
4. Casts rays in 16 directions to find clearest path
5. Moves character toward best direction
6. Adjusts camera as needed

## 🚀 Quick Start

### 1. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 2. Configure Minimap Location
```powershell
python configure_minimap.py
```

This interactive tool will:
- Help you mark where your minimap is on screen
- Capture your minimap
- Analyze path colors automatically
- Create `config_minimap.json`
- Show test visualization

### 3. Run the Navigator
```powershell
python minimap_navigator.py
```

## 📐 Configuration

The `configure_minimap.py` tool creates `config_minimap.json`:

```json
{
  "minimap_region": [50, 50, 200, 200],
  "path_color_lower": [150, 150, 150],
  "path_color_upper": [255, 255, 255],
  "obstacle_color_lower": [0, 0, 0],
  "obstacle_color_upper": [80, 80, 80],
  "movement_duration": 0.4,
  "scan_interval": 0.4,
  "look_ahead_distance": 30
}
```

### For Evil Lands Specifically:

Based on your screenshot, the minimap appears to be in the **top-right corner** (not top-left!). You'll need to:

1. Find exact coordinates using `configure_minimap.py`
2. Look for the **largest circle** on the top-right - that's your minimap
3. It's positioned **slightly down** from the very top edge
4. The minimap has a circular shape - the tool captures a square around it (okay)
5. Paths look **lighter/brighter** than obstacles
6. May need to filter out the circular border

**Typical coordinates for 1920x1080:**
```json
"minimap_region": [1670, 50, 200, 200]
```

## 🎯 Navigation Features

### 1. Narrow Path Following
Special algorithm that:
- Creates "skeleton" of path (centerline)
- Follows the center of narrow trails
- Perfect for Evil Lands pathways

### 2. Open Area Navigation
When not on a path:
- Casts 16 rays in all directions
- Finds direction with most open space
- Prefers forward movement

### 3. Obstacle Avoidance
- Detects dark areas as obstacles
- Adds safety margin around obstacles
- Checks immediate area before moving

### 4. Camera Adjustment
- Automatically rotates camera to face target direction
- Smooth turning
- Prevents getting stuck on walls

### 5. Stuck Detection
- Monitors if making progress
- Tries random direction if stuck
- Automatic recovery

## 🎮 Debug Mode

When running with `debug=True`, you see 3 windows:

1. **Minimap** - Raw minimap with player dot and direction arrow
2. **Paths** - Green = detected walkable areas
3. **Obstacles** - Red = detected obstacles

This helps you tune the color settings!

## ⚙️ Fine-Tuning for Evil Lands

### If paths not detected correctly:

1. **Check minimap images** created by configuration tool
2. **Adjust color thresholds** in `config_minimap.json`:
   - Lower `path_color_lower` if paths not detected
   - Raise `obstacle_color_upper` if obstacles not detected

3. **Example adjustments**:
```json
// More sensitive path detection
"path_color_lower": [120, 120, 120],

// More aggressive obstacle detection
"obstacle_color_upper": [100, 100, 100]
```

### If movement is jerky:

```json
"movement_duration": 0.5,  // Move longer
"scan_interval": 0.3        // React faster
```

### If bot gets stuck often:

```json
"look_ahead_distance": 40,  // Look further ahead
"stuck_threshold": 3        // React to stuck sooner
```

## 🔍 How Path Following Works

```
Minimap Image
    ↓
Color Analysis → Detect bright areas (paths)
              → Detect dark areas (obstacles)
    ↓
Path Skeleton → Find centerline of paths
    ↓
Ray Casting → Check 16 directions for clearness
    ↓
Best Direction → Calculate optimal angle
    ↓
Movement → Convert angle to WASD keys
    ↓
Camera Adjustment → Rotate if needed
```

## 📊 Console Output

```
✓ SAFE | NARROW PATH | Target: 45.2° | Frame: 10
✓ SAFE | NARROW PATH | Target: 43.8° | Frame: 11
⚠ CAUTION | OPEN AREA | Target: 90.5° | Frame: 12
⚠ No clear path detected - rotating to search...
✓ SAFE | NARROW PATH | Target: 25.1° | Frame: 13
```

- `✓ SAFE` = Clear immediate area
- `⚠ CAUTION` = Obstacles nearby
- `NARROW PATH` = Following a trail
- `OPEN AREA` = Navigating open space
- `Target: XX°` = Direction bot is moving (0°=North, 90°=East, etc.)

## 🆚 Minimap vs 3D Vision

| Feature | Minimap | 3D Vision |
|---------|---------|-----------|
| Accuracy | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| Speed | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Path Detection | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| Setup Difficulty | ⭐⭐ | ⭐⭐⭐⭐ |
| CPU Usage | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| Works at Night | ⭐⭐⭐⭐⭐ | ⭐ |
| Follows Trails | ⭐⭐⭐⭐⭐ | ⭐⭐ |

## ⚠️ Important Notes

1. **Minimap must be visible** - Don't hide it in game settings
2. **Circular minimap** - Tool handles this, captures square region
3. **UI elements** - Some UI on minimap is okay, it filters them out
4. **Different areas** - May need different color settings for different zones
5. **Test first** - Use debug mode to verify detection before long runs

## 🔧 Troubleshooting

### Bot moves in wrong directions
→ Run `configure_minimap.py` again to recalibrate colors

### Bot ignores obvious paths
→ Lower `path_color_lower` values (make detection more sensitive)

### Bot sees obstacles everywhere
→ Raise `obstacle_color_upper` values (less sensitive)

### Bot doesn't follow narrow trails
→ Check if path skeleton is working in debug view

### Camera spinning constantly
→ Increase `turn_threshold` to reduce sensitivity

## 📁 Files

- `minimap_navigator.py` - Main navigation bot (500+ lines)
- `configure_minimap.py` - Interactive setup tool
- `config_minimap.json` - Your configuration (auto-generated)
- `minimap_original.png` - Captured minimap sample
- `minimap_bright_areas.png` - Detected paths preview
- `minimap_dark_areas.png` - Detected obstacles preview

## 🎓 Advanced: Understanding the Algorithms

### Skeleton Algorithm
Reduces paths to single-pixel-wide lines, then follows the centerline. Perfect for narrow trails!

### Ray Casting
Shoots 16 virtual rays from player position, scores each direction based on path clearness.

### Direction Scoring
```python
score = (clear_pixels / total_pixels)
# 1.0 = completely clear
# 0.0 = completely blocked
```

### Camera Alignment
Automatically adjusts camera to face movement direction, prevents running sideways.

---

**This minimap approach is specifically designed for games like Evil Lands with visible minimaps showing paths! Much better than trying to understand 3D scenes! 🎯**
