# 🎯 Evil Lands Minimap Location Guide

## Where is the Minimap?

### Location
The minimap in Evil Lands is located in the **TOP-RIGHT corner** of the screen.

```
┌─────────────────────────────────────────────────────┐
│                                            ╔═══╗    │ ← Top edge
│                                            ║ M ║    │ ← Minimap (largest circle)
│                                            ║ A ║    │    slightly down from top
│                                            ║ P ║    │
│                                            ╚═══╝    │
│                                                     │
│                                                     │
│         [Game View - Character & World]            │
│                                                     │
│                                                     │
│  HP/Mana      [Abilities]         [Inventory]      │
└─────────────────────────────────────────────────────┘
```

## Key Characteristics

### Visual Identification:
1. ✅ **Largest circular element** on the top-right
2. ✅ Shows **map overview** with paths and terrain
3. ✅ Contains a **player indicator** (usually green dot or arrow)
4. ✅ Position: **Slightly down** from the very top edge (not at the absolute top)
5. ✅ Right side: **Close to the right edge** but not touching

## BlueStacks Resolution Examples

### 1920x1080 (Full HD)
```
Screen Width: 1920 pixels
Screen Height: 1080 pixels

Minimap Location:
┌─────────────────────────────────────────┐
│                              [1670, 50] │ ← Top-left of minimap
│                                    ●────┤
│                                    │  M │
│                                    │  A │
│                                    │  P │
│                                    ●────┤ ← Bottom-right [1870, 250]
│                              [200x200]  │
└─────────────────────────────────────────┘

Config: "minimap_region": [1670, 50, 200, 200]
```

### 1280x720 (HD)
```
Screen Width: 1280 pixels
Screen Height: 720 pixels

Minimap Location:
┌──────────────────────────────┐
│                    [1080,35] │ ← Top-left of minimap
│                         ●────┤
│                         │ M  │
│                         │ A  │
│                         ●────┤ ← Bottom-right [1230, 185]
│                   [150x150]  │
└──────────────────────────────┘

Config: "minimap_region": [1080, 35, 150, 150]
```

### 1600x900 (HD+)
```
Screen Width: 1600 pixels
Screen Height: 900 pixels

Minimap Location:
┌─────────────────────────────────┐
│                       [1350,40] │ ← Top-left of minimap
│                            ●────┤
│                            │ M  │
│                            │ A  │
│                            │ P  │
│                            ●────┤ ← Bottom-right [1525, 215]
│                      [175x175]  │
└─────────────────────────────────┘

Config: "minimap_region": [1350, 40, 175, 175]
```

## How to Find Exact Coordinates

### Step-by-Step:

1. **Open BlueStacks** with Evil Lands running
2. **Make sure minimap is visible** (check game settings)
3. **Run the configuration tool:**
   ```powershell
   python configure_minimap.py
   ```

4. **When prompted:**
   - Move mouse to the **top-left corner** of the minimap circle
   - Press Enter
   - Move mouse to the **bottom-right corner** of the minimap circle
   - Press Enter

5. **The tool will:**
   - Calculate exact coordinates
   - Capture the minimap
   - Analyze colors
   - Create `config_minimap.json` automatically

## Visual Markers to Look For

### What you'll see in the minimap:
- **Light/bright areas** = Walkable paths
- **Dark areas** = Obstacles, walls, cliffs
- **Your character** = Green dot or directional marker
- **Narrow lines** = Trails and pathways
- **Terrain features** = Trees, rocks, water

### Circular Border:
The minimap has a circular shape with a border. When you mark the region:
- **Include the entire circle** (the tool captures a square around it)
- The bot will process the entire square, which is fine
- The circular border won't interfere with path detection

## Common Mistakes

### ❌ Wrong Location:
- **NOT top-left corner** - Evil Lands uses top-right!
- **NOT centered at the very top** - It's slightly down

### ❌ Too Small:
- Make sure to capture the **entire minimap circle**
- If too small, bot won't see enough terrain
- Typical size: 150-200 pixels diameter

### ❌ Including Other UI:
- Don't include other buttons or icons
- Just the minimap circle itself
- Some overlap with borders is okay

## Verification

After running `configure_minimap.py`, check these files:

### 1. `minimap_original.png`
Should show:
- ✅ Clear circular minimap
- ✅ Visible paths and terrain
- ✅ No excessive other UI elements

### 2. `minimap_bright_areas.png`
Should show:
- ✅ White areas where paths are
- ✅ Clear separation between paths and obstacles

### 3. `minimap_dark_areas.png`
Should show:
- ✅ White areas where obstacles are
- ✅ Walls, blocked terrain highlighted

## If Coordinates are Wrong

### Symptoms:
- Bot captures wrong part of screen
- Debug window shows game UI instead of minimap
- No paths detected

### Solution:
```powershell
# Re-run configuration tool
python configure_minimap.py

# Be more precise with clicking:
# - Click exactly at minimap edges
# - Don't include surrounding UI
# - Make sure game window hasn't moved
```

## Manual Configuration

If you prefer to manually set coordinates:

1. **Take a screenshot** of your game (PrintScreen)
2. **Open in Paint** or image editor
3. **Note pixel coordinates** of minimap corners
4. **Edit `config_minimap.json`:**

```json
{
  "minimap_region": [left, top, width, height]
}
```

Where:
- `left` = X coordinate of top-left corner
- `top` = Y coordinate of top-left corner  
- `width` = minimap width in pixels
- `height` = minimap height in pixels

## Testing Your Configuration

After setting coordinates:

```powershell
# Test with live preview
python configure_minimap.py
# Choose 'y' when asked to test configuration

# Or run the bot with debug mode
python minimap_navigator.py
# Debug window will show if minimap is captured correctly
```

## Quick Reference

**Evil Lands Minimap:**
- Location: **Top-Right Corner**
- Appearance: **Largest Circle**
- Position: **Slightly Down** from top edge
- Contains: **Map, paths, player position**

**For BlueStacks 1920x1080:**
```json
"minimap_region": [1670, 50, 200, 200]
```

**Use the configuration tool for exact values:**
```powershell
python configure_minimap.py
```

---

**Having trouble? The debug window in `minimap_navigator.py` will show you exactly what's being captured!**
