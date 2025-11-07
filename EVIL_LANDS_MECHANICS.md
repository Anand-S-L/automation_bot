# Evil Lands Game Mechanics - Perception & Combat Guide

**Updated:** November 7, 2025  
**For:** State-of-the-Art RL Farming Agent

---

## 🎮 Game-Specific Mechanics

### Player UI (Top Left Corner)

#### HP Bar
- **Location:** Very top left
- **Color:** Red
- **Format:** Visual bar + numbers (e.g. "50/100")
- **Behavior:** 
  - Decreases when hit by enemies
  - Numbers update in real-time
  - Red bar shrinks from right to left

#### XP Bar (NOT Mana!)
- **Location:** Below HP bar at top left
- **Color:** Blue
- **Format:** Visual bar + numbers (e.g. "1234/5000")
- **Behavior:**
  - Increases when you kill enemies
  - Numbers go up with each kill
  - Bar fills from left to right
- **Important:** This is XP, NOT mana! No mana system in Evil Lands.

---

## 👹 Enemy Detection

### Enemy HP Bars

**Key Insight:** Enemy HP bars can be **RED, WHITE, or TRANSPARENT**!

#### Bar Colors:
1. **Red bars** - Common enemies
2. **White/gray bars** - Some enemy types
3. **Transparent bars** - Boss or special enemies

#### Critical Feature: Health Numbers
- **All bars have HP numbers in the middle**
- **Format:** "50/100" (current/max)
- **Location:** Center of the HP bar
- **This is the most reliable detection method!**

#### Detection Strategy:
```
1. Look for any horizontal bar above enemy
2. Check for numbers in format "XX/XX"
3. Parse current HP and max HP
4. Calculate percentage: (current/max) * 100
```

---

## ⚔️ Combat System

### Attack Mechanics

#### Red Attack Icon
- **Appears when:** You press spacebar to attack
- **Location:** Above the enemy being attacked
- **Color:** Bright red
- **Shape:** Icon/indicator
- **Duration:** Visible while enemy is alive

#### Combat Flow:
```
1. Target enemy (move close)
2. Press SPACEBAR to attack
   → Red attack icon appears above enemy
3. Keep spamming SPACEBAR
   → Continue until red icon disappears
4. Red icon disappears = Enemy is dead
5. Press 'B' to collect loot
```

#### Agent Learning Strategy:
- **Detect red attack icon** = Enemy is targeted and being attacked
- **Icon present** = Keep pressing spacebar (action 8)
- **Icon disappears** = Enemy dead, press B (action 9)
- **Reward:** +10 for kill, +5 for loot collected

---

## 💰 Loot Collection

### Loot Mechanic
- **After every kill:** Press 'B' key
- **Effect:** Automatically collects all nearby loot
- **Visual:** Items disappear from ground
- **Range:** Small radius around player

### Agent Strategy:
```
If enemy_dead (red icon gone):
    Press 'B' (collect_loot action)
    Wait 0.5 seconds
    Move to next enemy
```

---

## 🤖 Updated Perception System

### Health Detection Changes
```python
@dataclass
class HealthManaState:
    health_percentage: float   # From HP bar
    health_current: int        # From "50/100" OCR
    health_max: int            # From "50/100" OCR
    xp_percentage: float       # From XP bar (NOT mana)
    xp_current: int            # From XP numbers
    xp_max: int                # From XP numbers
    is_low_health: bool        # HP < 30%
    is_critical: bool          # HP < 15%
```

**Key Changes:**
- ❌ Removed `mana_percentage` (no mana in Evil Lands)
- ✅ Added `xp_percentage` (XP bar is blue, not mana)
- ✅ Added current/max HP numbers (OCR from "50/100")
- ✅ Added current/max XP numbers (OCR from "1234/5000")

### Enemy Detection Changes
```python
@dataclass
class Enemy:
    position: Tuple[int, int]
    hp_current: int            # NEW: From OCR "50/100"
    hp_max: int                # NEW: From OCR "50/100"
    hp_percentage: float       # Calculated from numbers
    has_attack_icon: bool      # NEW: Red icon detection
    is_targeted: bool          # True if attack icon present
    distance_score: float
```

**Key Changes:**
- ✅ Detect HP bars of ANY color (red, white, transparent)
- ✅ OCR to read HP numbers from bars
- ✅ Detect red attack icon (most reliable targeting method)
- ✅ `has_attack_icon` = Enemy is being attacked

---

## 🎯 Updated Action Space (16 Actions)

```python
0: 'up'                    # W key
1: 'down'                  # S key
2: 'left'                  # A key
3: 'right'                 # D key
4: 'up+left'               # W+A
5: 'up+right'              # W+D
6: 'down+left'             # S+A
7: 'down+right'            # S+D
8: 'attack'                # Spacebar - spam until icon gone
9: 'collect_loot'          # B key - after every kill
10: 'look_left'            # Numpad 4
11: 'look_right'           # Numpad 6
12: 'look_up'              # Numpad 8
13: 'look_down'            # Numpad 5
14: 'attack+move_forward'  # Spacebar + W
15: 'attack+collect'       # Spacebar + B combo
```

**New Actions:**
- **Action 9:** `collect_loot` - Press B after kill
- **Action 15:** `attack+collect` - Combo for efficiency

---

## 🏆 Updated Reward System

### Combat Rewards
```python
# Kill detection
if red_icon_disappears:  # Enemy died
    reward += 10.0
    
# Loot collection
if loot_collected:  # B pressed after kill
    reward += 5.0
    
# XP gain (detected from XP bar increase)
if xp_increased:
    reward += xp_gain * 0.1
    
# Damage taken (HP bar decreases)
if hp_decreased:
    reward -= hp_loss * 0.5
```

### Shaped Rewards
```python
# Attacking with red icon visible
if has_attack_icon and action == 8:  # Spamming spacebar
    reward += 0.05  # Encourage continued attack
    
# Collecting loot after kill
if red_icon_disappeared and action == 9:  # Pressed B
    reward += 2.0  # Bonus for timely collection
    
# Moving toward enemy with attack icon
if has_attack_icon and moving_closer:
    reward += 0.02
```

---

## 📝 Detection Configuration

### Color Ranges (HSV)

#### Player HP Bar (Red)
```python
hp_color_lower1 = [0, 100, 100]
hp_color_upper1 = [10, 255, 255]
hp_color_lower2 = [170, 100, 100]  # Red wraps in HSV
hp_color_upper2 = [180, 255, 255]
```

#### XP Bar (Blue)
```python
xp_color_lower = [100, 100, 100]
xp_color_upper = [130, 255, 255]
```

#### Enemy HP Bars (Multi-color)
```python
# Red bars
enemy_hp_red_lower = [0, 50, 50]
enemy_hp_red_upper = [10, 255, 255]

# White/gray bars
enemy_hp_white_lower = [0, 0, 150]
enemy_hp_white_upper = [180, 50, 255]

# Transparent/gray bars
enemy_hp_gray_lower = [0, 0, 100]
enemy_hp_gray_upper = [180, 100, 200]
```

#### Red Attack Icon
```python
attack_icon_lower = [0, 150, 150]
attack_icon_upper = [10, 255, 255]
```

---

## 🔧 OCR Configuration

### Reading HP Numbers

#### Preprocessing:
```python
# Extract bar region
bar_roi = screen[y:y+h, x:x+w]

# Convert to grayscale
gray = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2GRAY)

# Threshold (white text on black background)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Enlarge for better OCR
binary = cv2.resize(binary, (w*2, h*2))
```

#### Parsing:
```python
import re

# Run OCR
text = pytesseract.image_to_string(binary, config='--psm 7 digits/')

# Parse "50/100" format
match = re.search(r'(\d+)\s*/\s*(\d+)', text)
if match:
    current_hp = int(match.group(1))
    max_hp = int(match.group(2))
```

---

## 🎓 Learning Strategy

### Phase 1: Basic Combat (Episodes 1-100)
**Goal:** Learn to attack enemies

**Behavior:**
- Random exploration
- Discover spacebar = attack
- Learn red icon = target
- High death rate

**Rewards:**
- Small rewards for pressing spacebar near enemies
- Big reward when red icon disappears (kill)

### Phase 2: Loot Collection (Episodes 100-300)
**Goal:** Learn to collect loot after kills

**Behavior:**
- Attack enemies consistently
- Learn to press B after kills
- Associate icon disappearance with loot

**Rewards:**
- +10 for kill
- +5 for pressing B after kill
- Bonus for quick collection

### Phase 3: Efficient Farming (Episodes 300-500)
**Goal:** Optimize combat and collection

**Behavior:**
- Move toward enemies
- Spam spacebar efficiently
- Collect loot immediately
- Minimize damage taken

**Rewards:**
- Bonus for multi-kills
- Penalty for missing loot
- Reward for low damage taken

---

## 📊 Success Metrics

### Detection Accuracy
- HP bar detection: >90%
- XP bar detection: >95%
- Enemy HP number OCR: >80%
- Attack icon detection: >95%

### Combat Performance
- Kill rate: 10-30 per episode
- Loot collection rate: >80% of kills
- Damage efficiency: <30 HP lost per kill
- Attack accuracy: >60% spacebar presses hit

---

## 🐛 Common Issues

### "HP/XP not detected"
- Check bar regions in config
- Adjust color ranges
- Verify OCR is working
- Run test scripts

### "Attack icon not showing"
- Ensure you're pressing spacebar
- Check if enemy is in range
- Verify red color range
- Icon might be different shape

### "Loot not collected"
- Press B immediately after kill
- Check if B key is mapped correctly
- Verify loot is nearby
- May need to move closer

### "OCR not reading numbers"
- Install Tesseract or EasyOCR
- Check preprocessing (threshold)
- Enlarge ROI for better recognition
- Adjust PSM config

---

## ✅ Updated Checklist

### Perception System
- [x] HP bar (red, top left)
- [x] XP bar (blue, below HP) - NOT mana
- [x] HP/XP number OCR (e.g. "50/100")
- [x] Enemy HP bars (red, white, transparent)
- [x] Enemy HP number OCR
- [x] Red attack icon detection
- [x] Icon disappearance = kill

### Action System
- [x] Movement (WASD)
- [x] Attack (spacebar)
- [x] Loot collection (B key)
- [x] Camera controls (numpad)
- [x] Combo actions

### Reward System
- [x] Kill detection (icon disappears)
- [x] Loot collection (B pressed)
- [x] XP gain tracking
- [x] HP loss tracking
- [x] Shaped rewards for combat

---

## 🚀 Next Steps

1. **Test Updated Detectors**
   ```powershell
   python perception/health_detection.py
   python perception/enemy_detection.py
   ```

2. **Verify OCR Works**
   - Should read "50/100" from bars
   - Check console output

3. **Train Agent**
   ```powershell
   python rl_farming_agent.py
   # Option 1: Train 100 episodes
   ```

4. **Monitor Learning**
   - Watch for loot collection behavior
   - Check if agent spams spacebar
   - Verify B key is pressed after kills

---

**The agent should now:**
- ✅ Detect HP/XP correctly (no more mana confusion)
- ✅ Find enemies by HP numbers (any bar color)
- ✅ Detect red attack icon (reliable targeting)
- ✅ Learn to spam spacebar until icon gone
- ✅ Press B to collect loot after every kill
- ✅ Maximize kills and loot per episode

**Ready for Evil Lands! 🗡️⚔️💰**
