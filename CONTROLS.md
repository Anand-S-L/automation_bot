# Control Scheme Configuration

## Updated Controls for Evil Lands

Both navigation scripts have been updated to use Evil Lands control scheme:

### Movement Controls (Arrow Keys)
- **↑ (Up Arrow)** - Move Forward (North)
- **↓ (Down Arrow)** - Move Backward (South)
- **← (Left Arrow)** - Move Left (West)
- **→ (Right Arrow)** - Move Right (East)

### Diagonal Movement (Combined Keys)
- **↑ + →** - Move Forward-Right (Northeast)
- **↑ + ←** - Move Forward-Left (Northwest)
- **↓ + →** - Move Backward-Right (Southeast)
- **↓ + ←** - Move Backward-Left (Southwest)

### Camera Controls (Numpad - Free Look)
- **Numpad 4** - Look Left
- **Numpad 6** - Look Right
- **Numpad 8** - Look Up (not currently used, but supported)
- **Numpad 5** - Look Down (not currently used, but supported)

## How It Works

### Movement Direction Mapping
The bot analyzes the minimap/screen and determines an angle (0-360 degrees):
- **0° (North)** → Press `Up Arrow`
- **45° (Northeast)** → Press `Up + Right Arrows`
- **90° (East)** → Press `Right Arrow`
- **135° (Southeast)** → Press `Down + Right Arrows`
- **180° (South)** → Press `Down Arrow`
- **225° (Southwest)** → Press `Down + Left Arrows`
- **270° (West)** → Press `Left Arrow`
- **315° (Northwest)** → Press `Up + Left Arrows`

### Camera Rotation
When the bot needs to adjust the camera view:
- Calculates angle difference from current facing direction
- **Positive angle** (turn right) → Holds `Numpad 6`
- **Negative angle** (turn left) → Holds `Numpad 4`
- Duration scales with angle size (larger turns = longer key hold)

## Files Updated

### 1. `minimap_navigator.py`
Updated methods:
- `angle_to_direction()` - Converts angles to arrow key combinations
- `move_direction()` - Handles arrow key pressing/releasing
- `adjust_camera()` - Uses Numpad 4/6 instead of mouse movement
- `stop()` - Releases all arrow and numpad keys

### 2. `autonomous_navigator.py`
Updated methods:
- `MovementController.DIRECTIONS` - Changed from WASD to arrow keys
- `move()` - Handles arrow key combinations
- `rotate_camera()` - Uses Numpad 4/6 with scaled duration
- `stop()` - Releases all arrow and numpad keys

## Testing the Controls

### Before Running the Bot:

1. **Verify Game Controls**
   - Open Evil Lands
   - Go to Settings → Controls
   - Confirm:
     - Arrow keys = Movement
     - Numpad 4/6 = Camera rotation

2. **Manual Test**
   - In game, manually test each key
   - Make sure camera free look works with numpad
   - Check that arrow keys move character in correct directions

3. **Check Numpad State**
   - Make sure NumLock is ON
   - Test numpad keys work in game

### Running the Bot:

```powershell
# Minimap-based navigation (RECOMMENDED)
python minimap_navigator.py

# 3D vision-based navigation
python autonomous_navigator.py
```

## Troubleshooting

### Bot doesn't move
- ✓ Check that arrow keys control movement in game
- ✓ Verify game window is in focus
- ✓ Make sure NumLock is ON

### Camera doesn't rotate
- ✓ Check that Numpad 4/6 control camera in game
- ✓ Verify "Free Look" mode is enabled
- ✓ Test numpad keys manually in game first

### Movement is diagonal when it should be straight
- ✓ Keys might be getting stuck
- ✓ Stop the bot (Ctrl+C) to release all keys
- ✓ Check for key conflicts in game settings

### Wrong direction
- ✓ Verify camera is facing the right direction initially
- ✓ Bot assumes 0° = North (forward)
- ✓ May need calibration if game has different orientation

## Configuration Options

You can adjust sensitivity in config files:

### `config_minimap.json`:
```json
{
  "movement_duration": 0.4,      // How long to hold arrow keys
  "turn_threshold": 15.0,        // Min angle before camera rotation
  "scan_interval": 0.4           // Time between movements
}
```

### Faster Movement:
```json
{
  "movement_duration": 0.25,     // Shorter key presses
  "scan_interval": 0.3           // React quicker
}
```

### Smoother Rotation:
```json
{
  "turn_threshold": 10.0         // Rotate with smaller angle differences
}
```

## Advanced: Vertical Camera Control

If you need vertical camera adjustment (looking up/down):

The code supports Numpad 8 (up) and 5 (down), but doesn't use them by default. To enable:

In `minimap_navigator.py`, add to `adjust_camera()` method:

```python
# Add vertical adjustment (if needed)
if vertical_angle > threshold:
    pyautogui.keyDown('num8')  # Look up
    time.sleep(0.2)
    pyautogui.keyUp('num8')
elif vertical_angle < -threshold:
    pyautogui.keyDown('num5')  # Look down
    time.sleep(0.2)
    pyautogui.keyUp('num5')
```

## Key Release Safety

Both scripts include automatic key release on shutdown:
- Press `Ctrl+C` to stop
- All keys automatically released
- Prevents stuck keys
- Safe exit from any state

## Summary

✅ Movement: Arrow Keys (8-directional)  
✅ Camera: Numpad 4 (left), 6 (right)  
✅ Vertical Look: Numpad 8 (up), 5 (down) - supported but not used  
✅ Safe shutdown: Ctrl+C releases all keys  
✅ Works with both minimap and 3D vision scripts  

Perfect for Evil Lands control scheme! 🎮
