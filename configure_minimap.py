"""
Configuration helper for minimap-based navigation
Use this to find your minimap location and calibrate path colors
"""

import cv2
import numpy as np
import time
from mss import mss
import pyautogui
import json


def find_minimap_region():
    """Interactive tool to locate minimap on screen"""
    print("=" * 60)
    print("MINIMAP REGION FINDER")
    print("=" * 60)
    print("\nLet's find your minimap location!")
    print("\n⚠ IMPORTANT: In Evil Lands, the minimap is on the TOP-RIGHT corner")
    print("   Look for the LARGEST CIRCLE on the top-right")
    print("   It's positioned SLIGHTLY DOWN from the very top edge\n")
    
    print("1. Move mouse to TOP-LEFT corner of the minimap circle")
    print("   Press Enter when ready...")
    input()
    top_left = pyautogui.position()
    print(f"   ✓ Top-Left: ({top_left.x}, {top_left.y})")
    
    print("\n2. Move mouse to BOTTOM-RIGHT corner of the minimap circle")
    print("   Press Enter when ready...")
    input()
    bottom_right = pyautogui.position()
    print(f"   ✓ Bottom-Right: ({bottom_right.x}, {bottom_right.y})")
    
    left = top_left.x
    top = top_left.y
    width = bottom_right.x - top_left.x
    height = bottom_right.y - top_left.y
    
    return (left, top, width, height)


def calibrate_colors(minimap_region):
    """Capture minimap and help identify path/obstacle colors"""
    print("\n" + "=" * 60)
    print("COLOR CALIBRATION")
    print("=" * 60)
    print("\n3. Switch to game window...")
    print("   Make sure minimap is clearly visible")
    print("   Capturing in 3 seconds...")
    time.sleep(3)
    
    # Capture minimap
    sct = mss()
    region = {
        "left": minimap_region[0],
        "top": minimap_region[1],
        "width": minimap_region[2],
        "height": minimap_region[3]
    }
    screenshot = sct.grab(region)
    minimap = np.array(screenshot)
    minimap = cv2.cvtColor(minimap, cv2.COLOR_BGRA2BGR)
    
    # Save original
    cv2.imwrite('minimap_original.png', minimap)
    print("\n✓ Saved minimap as 'minimap_original.png'")
    
    # Analyze color ranges
    print("\n4. Analyzing minimap colors...")
    
    # Create HSV version for better analysis
    hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
    
    # Analyze brightness
    gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    std_brightness = np.std(gray)
    
    print(f"\n   Average brightness: {mean_brightness:.1f}")
    print(f"   Brightness variation: {std_brightness:.1f}")
    
    # Create sample masks with different thresholds
    print("\n5. Creating sample masks...")
    
    # Light areas (potential paths)
    _, bright_mask = cv2.threshold(gray, mean_brightness + std_brightness * 0.3, 255, cv2.THRESH_BINARY)
    cv2.imwrite('minimap_bright_areas.png', bright_mask)
    
    # Dark areas (potential obstacles)
    _, dark_mask = cv2.threshold(gray, mean_brightness - std_brightness * 0.3, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite('minimap_dark_areas.png', dark_mask)
    
    print("   ✓ Saved 'minimap_bright_areas.png' (potential paths)")
    print("   ✓ Saved 'minimap_dark_areas.png' (potential obstacles)")
    
    # Interactive color picker
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            bgr = minimap[y, x]
            hsv_val = hsv[y, x]
            print(f"\n   Clicked pixel - BGR: {bgr}, HSV: {hsv_val}")
    
    print("\n6. Click on different minimap areas to see their colors:")
    print("   - Click on PATHS (walkable areas)")
    print("   - Click on OBSTACLES (walls, blocked areas)")
    print("   - Press ESC when done")
    
    # Show minimap with interactive color picker
    minimap_large = cv2.resize(minimap, (minimap.shape[1] * 3, minimap.shape[0] * 3))
    cv2.namedWindow('Minimap - Click to get colors')
    cv2.setMouseCallback('Minimap - Click to get colors', mouse_callback)
    cv2.imshow('Minimap - Click to get colors', minimap_large)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Suggest color ranges based on analysis
    bright_threshold = int(mean_brightness + std_brightness * 0.3)
    dark_threshold = int(mean_brightness - std_brightness * 0.5)
    
    suggested_config = {
        "path_color_lower": [bright_threshold, bright_threshold, bright_threshold],
        "path_color_upper": [255, 255, 255],
        "obstacle_color_lower": [0, 0, 0],
        "obstacle_color_upper": [dark_threshold, dark_threshold, dark_threshold]
    }
    
    return suggested_config


def create_config(minimap_region, color_config):
    """Create configuration file"""
    config = {
        "minimap_region": list(minimap_region),
        "path_color_lower": color_config["path_color_lower"],
        "path_color_upper": color_config["path_color_upper"],
        "obstacle_color_lower": color_config["obstacle_color_lower"],
        "obstacle_color_upper": color_config["obstacle_color_upper"],
        "movement_duration": 0.4,
        "scan_interval": 0.4,
        "look_ahead_distance": 30,
        "turn_threshold": 15.0,
        "stuck_threshold": 5,
        
        "_comment1": "minimap_region: [left, top, width, height] of minimap on screen",
        "_comment2": "path_color_lower/upper: BGR color range for walkable paths",
        "_comment3": "obstacle_color_lower/upper: BGR color range for obstacles",
        "_comment4": "movement_duration: how long to move in each direction",
        "_comment5": "scan_interval: time between minimap scans",
        "_comment6": "look_ahead_distance: pixels to look ahead on minimap",
        "_comment7": "turn_threshold: degrees before adjusting camera",
        "_comment8": "stuck_threshold: frames before considering stuck"
    }
    
    with open('config_minimap.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("\n" + "=" * 60)
    print("✓ CONFIGURATION SAVED")
    print("=" * 60)
    print(f"\nMinimap Region: {minimap_region}")
    print(f"Path Colors: {color_config['path_color_lower']} to {color_config['path_color_upper']}")
    print(f"Obstacle Colors: {color_config['obstacle_color_lower']} to {color_config['obstacle_color_upper']}")
    print("\n✓ Saved as 'config_minimap.json'")
    print("\nNext steps:")
    print("  1. Review the generated minimap images")
    print("  2. If colors look wrong, manually edit config_minimap.json")
    print("  3. Run: python minimap_navigator.py")
    print("=" * 60)


def main():
    """Main configuration flow"""
    print("\n" + "=" * 60)
    print("EVIL LANDS MINIMAP NAVIGATOR - SETUP")
    print("=" * 60)
    print("\nThis tool will help you configure minimap-based navigation.")
    print("Make sure Evil Lands is running and minimap is visible!")
    print("\nPress Enter to continue...")
    input()
    
    # Step 1: Find minimap region
    minimap_region = find_minimap_region()
    
    # Step 2: Calibrate colors
    color_config = calibrate_colors(minimap_region)
    
    # Step 3: Create config file
    create_config(minimap_region, color_config)
    
    # Step 4: Test visualization
    print("\n\nWould you like to test the configuration? (y/n): ", end='')
    if input().lower() == 'y':
        test_configuration(minimap_region, color_config)


def test_configuration(minimap_region, color_config):
    """Test the configuration with live preview"""
    print("\n" + "=" * 60)
    print("TESTING CONFIGURATION")
    print("=" * 60)
    print("\nShowing live minimap analysis...")
    print("Press 'q' to stop")
    print("\nSwitch to game in 3 seconds...")
    time.sleep(3)
    
    sct = mss()
    region = {
        "left": minimap_region[0],
        "top": minimap_region[1],
        "width": minimap_region[2],
        "height": minimap_region[3]
    }
    
    try:
        while True:
            # Capture minimap
            screenshot = sct.grab(region)
            minimap = np.array(screenshot)
            minimap = cv2.cvtColor(minimap, cv2.COLOR_BGRA2BGR)
            
            # Apply path detection
            gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
            path_lower = color_config['path_color_lower'][0]
            _, path_mask = cv2.threshold(gray, path_lower, 255, cv2.THRESH_BINARY)
            
            # Apply obstacle detection
            obstacle_upper = color_config['obstacle_color_upper'][0]
            _, obstacle_mask = cv2.threshold(gray, obstacle_upper, 255, cv2.THRESH_BINARY_INV)
            
            # Create visualization
            path_overlay = cv2.cvtColor(path_mask, cv2.COLOR_GRAY2BGR)
            path_overlay[:, :, 1] = path_mask
            
            obstacle_overlay = cv2.cvtColor(obstacle_mask, cv2.COLOR_GRAY2BGR)
            obstacle_overlay[:, :, 2] = obstacle_mask
            
            # Draw player position
            center = (minimap.shape[1] // 2, minimap.shape[0] // 2)
            cv2.circle(minimap, center, 5, (0, 255, 0), -1)
            
            # Combine
            combined = np.hstack([minimap, path_overlay, obstacle_overlay])
            combined = cv2.resize(combined, (combined.shape[1] * 2, combined.shape[0] * 2))
            
            # Labels
            cv2.putText(combined, "Minimap", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(combined, "Paths (Green)", (minimap.shape[1] * 2 + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(combined, "Obstacles (Red)", (minimap.shape[1] * 4 + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Configuration Test - Press Q to exit', combined)
            
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        print("\n✓ Test complete!")


if __name__ == "__main__":
    main()
