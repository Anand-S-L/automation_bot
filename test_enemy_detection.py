"""
Test Enemy Detection - Debug why 0 enemies detected
"""

import cv2
import numpy as np
import mss
import json
from perception.enemy_detection import EnemyDetector

# Load config
with open('config_spatial.json', 'r') as f:
    config = json.load(f)

# Ensure enemy_detection has game_region and minimap_region
enemy_config = config.get('enemy_detection', {})
if 'game_region' not in enemy_config:
    enemy_config['game_region'] = config.get('game_region', [0, 0, 1920, 1080])
if 'minimap_region' not in enemy_config:
    enemy_config['minimap_region'] = config.get('minimap_region', [1670, 50, 200, 200])

# Initialize detector
print("Initializing enemy detector...")
print(f"Config: {enemy_config}")
detector = EnemyDetector(enemy_config)

print("\nStarting detection test...")
print("Press 'q' to quit")
print("Press 's' to save debug images")
print("Press '1' to test HP bars only")
print("Press '2' to test attack icons only")
print("Press '3' to test minimap only")
print("Press '4' to test hybrid mode")
print("Press 'd' to toggle detection region visualization")
print("\n")

sct = mss.mss()
monitor = sct.monitors[1]

show_debug = True
frame_count = 0

while True:
    # Capture screen
    screenshot = sct.grab(monitor)
    screen = np.array(screenshot)
    screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
    
    # Get minimap region
    mx, my, mw, mh = config['minimap_region']
    minimap = screen[my:my+mh, mx:mx+mw]
    
    # Detect enemies
    state = detector.detect(screen, minimap)
    
    # Visualize
    vis = detector.visualize(screen, state)
    
    # Draw detection region if enabled
    if show_debug and detector.use_detection_region:
        height, width = screen.shape[:2]
        region_width = int(width * detector.detection_region_percent)
        region_height = int(height * detector.detection_region_percent)
        offset_x = (width - region_width) // 2
        offset_y = (height - region_height) // 2
        
        # Draw detection region boundary
        cv2.rectangle(vis, (offset_x, offset_y), 
                     (offset_x + region_width, offset_y + region_height),
                     (255, 255, 0), 3)
        cv2.putText(vis, "Detection Region", (offset_x + 10, offset_y + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
    
    # Draw minimap region
    cv2.rectangle(vis, (mx, my), (mx+mw, my+mh), (0, 255, 255), 2)
    cv2.putText(vis, "Minimap", (mx, my-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Show debug info
    info_y = 100
    debug_info = [
        f"Frame: {frame_count}",
        f"Detection Method: {detector.detection_method}",
        f"Enemies: {state.enemy_count}",
        f"Has Target: {state.has_target}",
        f"Detection Region: {'ON' if detector.use_detection_region else 'OFF'} ({detector.detection_region_percent*100:.0f}%)",
        f"Minimap Prefilter: {'ON' if detector.minimap_prefilter else 'OFF'}",
        f"Screen: {screen.shape[1]}x{screen.shape[0]}"
    ]
    
    for i, info in enumerate(debug_info):
        cv2.putText(vis, info, (10, info_y + i*25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Resize for display
    vis = cv2.resize(vis, (1280, 720))
    minimap_big = cv2.resize(minimap, (400, 400))
    
    cv2.imshow('Enemy Detection Test', vis)
    cv2.imshow('Minimap', minimap_big)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite(f'debug_screen_{frame_count}.png', screen)
        cv2.imwrite(f'debug_vis_{frame_count}.png', vis)
        cv2.imwrite(f'debug_minimap_{frame_count}.png', minimap)
        print(f"Saved debug images for frame {frame_count}")
    elif key == ord('1'):
        detector.detection_method = 'hp_bars'
        print("Switched to HP bars only")
    elif key == ord('2'):
        detector.detection_method = 'attack_icon'
        print("Switched to attack icons only")
    elif key == ord('3'):
        detector.detection_method = 'minimap'
        print("Switched to minimap only")
    elif key == ord('4'):
        detector.detection_method = 'hybrid'
        print("Switched to hybrid mode")
    elif key == ord('d'):
        show_debug = not show_debug
        print(f"Detection region visualization: {'ON' if show_debug else 'OFF'}")
    
    frame_count += 1

cv2.destroyAllWindows()
print("\nTest complete!")
