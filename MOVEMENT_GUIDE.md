# 🎮 Movement Configuration Guide

## Problem: Stop-Start Movement or Not Following Paths

If your bot has jerky, stop-start movement or doesn't follow minimap paths, adjust these settings:

---

## ⚡ Quick Fix

Edit your `config_minimap.json` with these improved values:

```json
{
  "movement_duration": 0.8,
  "scan_interval": 0.2,
  "look_ahead_distance": 40,
  "turn_threshold": 20.0,
  "stuck_threshold": 10,
  "continuous_movement": true,
  "path_color_lower": [80, 80, 80]
}
```

---

## 🔧 Setting Explanations

### `movement_duration` (Default: 0.8)
**What it does:** How long to hold movement keys

**For smooth movement:**
```json
"movement_duration": 0.8  // Longer = smoother but slower turns
```

**For faster/responsive:**
```json
"movement_duration": 0.5  // Shorter = quicker reactions
```

**For very smooth exploration:**
```json
"movement_duration": 1.0  // Longest = most fluid motion
```

---

### `scan_interval` (Default: 0.2)
**What it does:** Time between minimap scans

**For responsive movement:**
```json
"scan_interval": 0.2  // Fast updates = quick reactions
```

**For slower/stable:**
```json
"scan_interval": 0.5  // Slower = less CPU, more stable
```

**Balance (recommended):**
```json
"scan_interval": 0.3  // Good middle ground
```

---

### `continuous_movement` (Default: true)
**What it does:** Keep keys pressed between scans for fluid motion

```json
"continuous_movement": true   // Smooth continuous movement ✅
"continuous_movement": false  // Stop-start movement (old behavior)
```

**When to use false:**
- If character moves too fast
- Need precise stop-start control
- Game has input lag

---

### `look_ahead_distance` (Default: 40)
**What it does:** How far ahead to look on minimap (pixels)

**For narrow paths:**
```json
"look_ahead_distance": 30  // Shorter = tighter following
```

**For open areas:**
```json
"look_ahead_distance": 50  // Longer = plan further ahead
```

**For mixed terrain (recommended):**
```json
"look_ahead_distance": 40  // Good balance
```

---

### `turn_threshold` (Default: 20.0)
**What it does:** Minimum angle difference before camera rotation

**For smooth movement (less turning):**
```json
"turn_threshold": 25.0  // Only turn for big direction changes
```

**For precise following:**
```json
"turn_threshold": 15.0  // Turn more often, tighter control
```

**For very smooth (recommended):**
```json
"turn_threshold": 20.0  // Good balance
```

---

### `path_color_lower` (Default: [80, 80, 80])
**What it does:** Minimum brightness for path detection

**If paths not detected:**
```json
"path_color_lower": [60, 60, 60]  // Lower = more sensitive
```

**If too many false paths:**
```json
"path_color_lower": [100, 100, 100]  // Higher = more strict
```

**Recommended for Evil Lands:**
```json
"path_color_lower": [80, 80, 80]  // Works for most minimaps
```

---

### `stuck_threshold` (Default: 10)
**What it does:** Frames before triggering stuck recovery

**For patient navigation:**
```json
"stuck_threshold": 15  // Wait longer before unstuck action
```

**For aggressive unstuck:**
```json
"stuck_threshold": 5  // React quickly to being stuck
```

---

## 🎯 Preset Configurations

### Smooth Exploration (Recommended)
```json
{
  "movement_duration": 0.8,
  "scan_interval": 0.2,
  "continuous_movement": true,
  "turn_threshold": 20.0,
  "look_ahead_distance": 40,
  "stuck_threshold": 10
}
```

### Fast & Responsive
```json
{
  "movement_duration": 0.5,
  "scan_interval": 0.15,
  "continuous_movement": true,
  "turn_threshold": 15.0,
  "look_ahead_distance": 35,
  "stuck_threshold": 7
}
```

### Stable & Cautious
```json
{
  "movement_duration": 1.0,
  "scan_interval": 0.3,
  "continuous_movement": true,
  "turn_threshold": 25.0,
  "look_ahead_distance": 50,
  "stuck_threshold": 15
}
```

### Precise Path Following
```json
{
  "movement_duration": 0.6,
  "scan_interval": 0.25,
  "continuous_movement": true,
  "turn_threshold": 12.0,
  "look_ahead_distance": 30,
  "stuck_threshold": 8
}
```

---

## 🐛 Troubleshooting Movement Issues

### Issue: Character moves too fast
**Solution:**
```json
"movement_duration": 0.5,  // Reduce
"scan_interval": 0.3       // Increase
```

### Issue: Character moves too slow
**Solution:**
```json
"movement_duration": 0.8,  // Keep or increase
"scan_interval": 0.15      // Decrease for faster updates
```

### Issue: Jerky stop-start movement
**Solution:**
```json
"continuous_movement": true,  // Enable smooth movement
"movement_duration": 0.8,     // Increase
"scan_interval": 0.2          // Decrease
```

### Issue: Not following paths
**Solutions:**

1. **Check path detection:**
   - Look at `minimap_bright_areas.png`
   - Should show paths in white
   
2. **Lower path threshold:**
   ```json
   "path_color_lower": [60, 60, 60]
   ```

3. **Re-run configuration:**
   ```powershell
   python configure_minimap.py
   ```

### Issue: Hitting walls frequently
**Solution:**
```json
"look_ahead_distance": 50,  // Look further
"turn_threshold": 15.0      // Turn more responsively
```

### Issue: Camera spinning too much
**Solution:**
```json
"turn_threshold": 30.0  // Only turn for large changes
```

### Issue: Gets stuck often
**Solution:**
```json
"stuck_threshold": 5,       // React faster
"look_ahead_distance": 45   // Plan better
```

---

## 🎮 BlueStacks-Specific Tips

### Input Lag
If BlueStacks has input lag:
```json
{
  "movement_duration": 0.7,
  "scan_interval": 0.3,
  "continuous_movement": false  // Disable if keys stick
}
```

### High Performance
For powerful PCs:
```json
{
  "movement_duration": 0.6,
  "scan_interval": 0.15,  // Very fast updates
  "continuous_movement": true
}
```

### Lower Performance
For slower PCs:
```json
{
  "movement_duration": 0.8,
  "scan_interval": 0.4,   // Slower updates
  "continuous_movement": true
}
```

---

## 🔄 Testing Your Settings

### Quick Test:
```powershell
python minimap_navigator.py
```

**Watch for:**
- ✅ Smooth continuous movement
- ✅ Follows bright paths on minimap
- ✅ Avoids dark obstacles
- ✅ Doesn't get stuck frequently
- ✅ Camera adjusts smoothly

**Run for 1-2 minutes** in different terrain types.

---

## 📊 Console Output

Good movement shows:
```
✓ SAFE | NARROW PATH | Target: 45.2° | Clearness: 0.85 | Frame: 10
✓ SAFE | NARROW PATH | Target: 43.8° | Clearness: 0.87 | Frame: 11
✓ SAFE | OPEN AREA | Target: 90.5° | Clearness: 0.72 | Frame: 12
```

**Clearness values:**
- `> 0.7` = Good path detected
- `0.5-0.7` = Moderate path
- `< 0.5` = Uncertain/obstacles

---

## 💡 Advanced: Fine-Tuning

### For Different Terrain Types:

**Open Fields:**
```json
{
  "look_ahead_distance": 50,
  "turn_threshold": 25.0,
  "movement_duration": 0.9
}
```

**Narrow Paths/Trails:**
```json
{
  "look_ahead_distance": 30,
  "turn_threshold": 12.0,
  "movement_duration": 0.6
}
```

**Crowded Areas:**
```json
{
  "look_ahead_distance": 35,
  "turn_threshold": 15.0,
  "movement_duration": 0.5,
  "scan_interval": 0.15
}
```

---

## 🔧 Real-Time Adjustments

You can edit `config_minimap.json` while the bot is running:
1. Stop the bot (Ctrl+C)
2. Edit config file
3. Save changes
4. Restart bot

Changes take effect immediately!

---

## ✅ Recommended Starting Point

Copy this to your `config_minimap.json`:

```json
{
  "minimap_region": [1670, 50, 200, 200],
  "path_color_lower": [80, 80, 80],
  "path_color_upper": [255, 255, 255],
  "obstacle_color_lower": [0, 0, 0],
  "obstacle_color_upper": [80, 80, 80],
  "movement_duration": 0.8,
  "scan_interval": 0.2,
  "look_ahead_distance": 40,
  "turn_threshold": 20.0,
  "stuck_threshold": 10,
  "continuous_movement": true
}
```

Then adjust based on your experience!

---

**Good luck with smooth navigation! 🚀**
