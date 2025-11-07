"""
Autonomous Game Navigator
A framework for autonomous navigation in open-world games with obstacle avoidance.
Educational purposes only.
"""

import cv2
import numpy as np
import time
from mss import mss
import pyautogui
from dataclasses import dataclass
from typing import Tuple, List
import json


@dataclass
class BotConfig:
    """Configuration for the navigation bot"""
    screen_region: Tuple[int, int, int, int] = (0, 0, 800, 600)  # (left, top, width, height)
    movement_duration: float = 0.3  # How long to hold movement key
    scan_interval: float = 0.5  # Time between scans
    obstacle_threshold: int = 50  # Darkness threshold for obstacles
    turn_angle: float = 45  # Degrees to turn when obstacle detected
    exploration_bias: float = 0.7  # 0-1, higher = prefer unexplored areas
    
    @classmethod
    def from_file(cls, filepath: str):
        """Load configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
                return cls(**config_dict)
        except FileNotFoundError:
            return cls()


class ScreenCapture:
    """Handles screen capture for game analysis"""
    
    def __init__(self, region: Tuple[int, int, int, int]):
        self.sct = mss()
        self.region = {
            "left": region[0],
            "top": region[1],
            "width": region[2],
            "height": region[3]
        }
    
    def capture(self) -> np.ndarray:
        """Capture screen and return as numpy array"""
        screenshot = self.sct.grab(self.region)
        img = np.array(screenshot)
        # Convert BGRA to BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    
    def capture_gray(self) -> np.ndarray:
        """Capture screen in grayscale"""
        img = self.capture()
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


class ObstacleDetector:
    """Detects obstacles and safe paths in game screen"""
    
    def __init__(self, obstacle_threshold: int = 50):
        self.obstacle_threshold = obstacle_threshold
    
    def detect_obstacles(self, image: np.ndarray) -> np.ndarray:
        """
        Detect obstacles in the image
        Returns a binary mask where 1 = obstacle, 0 = safe
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Method 1: Darkness detection (dark areas = obstacles)
        _, dark_mask = cv2.threshold(gray, self.obstacle_threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Method 2: Edge detection (strong edges = obstacles)
        edges = cv2.Canny(gray, 50, 150)
        
        # Method 3: Color-based detection (can be customized per game)
        # For example, detect water (blue), lava (red), etc.
        if len(image.shape) == 3:
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            # Example: detect blue water areas
            blue_lower = np.array([90, 50, 50])
            blue_upper = np.array([130, 255, 255])
            water_mask = cv2.inRange(hsv, blue_lower, blue_upper)
        else:
            water_mask = np.zeros_like(gray)
        
        # Combine detection methods
        combined = cv2.bitwise_or(dark_mask, edges)
        combined = cv2.bitwise_or(combined, water_mask)
        
        # Apply morphological operations to clean up noise
        kernel = np.ones((5, 5), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)
        
        return combined
    
    def analyze_directions(self, obstacle_mask: np.ndarray, num_sectors: int = 8) -> List[float]:
        """
        Analyze different directions for obstacle density
        Returns list of clearness scores (0-1) for each direction
        0 = completely blocked, 1 = completely clear
        """
        height, width = obstacle_mask.shape
        center_x, center_y = width // 2, height // 2
        
        # Focus on the area in front of the player (bottom half of screen)
        roi = obstacle_mask[center_y:, :]
        
        sector_width = width // num_sectors
        clearness_scores = []
        
        for i in range(num_sectors):
            start_x = i * sector_width
            end_x = (i + 1) * sector_width
            sector = roi[:, start_x:end_x]
            
            # Calculate percentage of clear pixels
            total_pixels = sector.size
            obstacle_pixels = np.count_nonzero(sector)
            clearness = 1.0 - (obstacle_pixels / total_pixels)
            clearness_scores.append(clearness)
        
        return clearness_scores
    
    def find_best_direction(self, clearness_scores: List[float]) -> int:
        """
        Find the best direction to move based on clearness scores
        Returns index of best sector (0 = far left, middle = center, etc.)
        """
        if not clearness_scores:
            return len(clearness_scores) // 2  # Default to center
        
        # Prefer forward direction with slight randomization
        center_idx = len(clearness_scores) // 2
        
        # Add bias toward center (forward movement)
        weighted_scores = clearness_scores.copy()
        for i in range(len(weighted_scores)):
            distance_from_center = abs(i - center_idx)
            # Reduce weight for extreme angles
            weighted_scores[i] *= (1.0 - distance_from_center * 0.1)
        
        # Find best direction
        best_idx = np.argmax(weighted_scores)
        return best_idx


class MovementController:
    """Controls character movement through input simulation"""
    
    # Direction mapping (8 directions) using arrow keys
    DIRECTIONS = {
        0: ['left'],           # Far left
        1: ['up', 'left'],     # Forward-left
        2: ['up'],             # Forward
        3: ['up', 'right'],    # Forward-right
        4: ['right'],          # Far right
        5: ['down', 'right'],  # Back-right
        6: ['down'],           # Back
        7: ['down', 'left'],   # Back-left
    }
    
    def __init__(self, movement_duration: float = 0.3):
        self.movement_duration = movement_duration
        self.current_direction = 2  # Start facing forward
        pyautogui.PAUSE = 0.05  # Reduce delay between commands
    
    def move(self, direction_idx: int):
        """Move in specified direction using arrow keys"""
        keys = self.DIRECTIONS.get(direction_idx, ['up'])
        
        # Press and hold the keys
        for key in keys:
            pyautogui.keyDown(key)
        
        time.sleep(self.movement_duration)
        
        # Release the keys
        for key in keys:
            pyautogui.keyUp(key)
        
        self.current_direction = direction_idx
    
    def rotate_camera(self, angle: float):
        """Rotate camera view using numpad keys (4=left, 6=right)"""
        # Use numpad 4 for left, 6 for right
        if angle > 0:
            # Rotate right
            pyautogui.keyDown('num6')
            time.sleep(abs(angle) / 180)  # Scale duration by angle
            pyautogui.keyUp('num6')
        else:
            # Rotate left
            pyautogui.keyDown('num4')
            time.sleep(abs(angle) / 180)
            pyautogui.keyUp('num4')
    
    def stop(self):
        """Release all movement keys"""
        all_keys = ['up', 'down', 'left', 'right', 'num4', 'num6', 'num8', 'num5']
        for key in all_keys:
            try:
                pyautogui.keyUp(key)
            except:
                pass


class AutonomousNavigator:
    """Main autonomous navigation system"""
    
    def __init__(self, config: BotConfig):
        self.config = config
        self.screen_capture = ScreenCapture(config.screen_region)
        self.obstacle_detector = ObstacleDetector(config.obstacle_threshold)
        self.movement_controller = MovementController(config.movement_duration)
        self.running = False
        self.debug_mode = False
    
    def start(self, debug: bool = False):
        """Start autonomous navigation"""
        self.running = True
        self.debug_mode = debug
        
        print("Autonomous Navigator Starting...")
        print(f"Screen Region: {self.config.screen_region}")
        print("Press Ctrl+C to stop")
        print("\nWaiting 3 seconds before starting - switch to game window...")
        time.sleep(3)
        
        try:
            self._navigation_loop()
        except KeyboardInterrupt:
            print("\nStopping navigator...")
        finally:
            self.stop()
    
    def _navigation_loop(self):
        """Main navigation loop"""
        while self.running:
            try:
                # Capture current screen
                screen = self.screen_capture.capture()
                
                # Detect obstacles
                obstacle_mask = self.obstacle_detector.detect_obstacles(screen)
                
                # Analyze directions
                clearness_scores = self.obstacle_detector.analyze_directions(obstacle_mask)
                
                # Find best direction
                best_direction = self.obstacle_detector.find_best_direction(clearness_scores)
                
                # Display debug info
                if self.debug_mode:
                    self._show_debug(screen, obstacle_mask, clearness_scores, best_direction)
                
                # Make movement decision
                if clearness_scores[best_direction] < 0.3:
                    # Too many obstacles, turn around
                    print("⚠ Heavy obstacles detected - rotating...")
                    self.movement_controller.rotate_camera(45)
                    time.sleep(0.2)
                else:
                    # Move in best direction
                    direction_name = self._get_direction_name(best_direction)
                    print(f"✓ Moving {direction_name} (clearness: {clearness_scores[best_direction]:.2f})")
                    self.movement_controller.move(best_direction)
                
                # Wait before next scan
                time.sleep(self.config.scan_interval)
                
            except Exception as e:
                print(f"Error in navigation loop: {e}")
                time.sleep(1)
    
    def _show_debug(self, screen: np.ndarray, obstacle_mask: np.ndarray, 
                    clearness_scores: List[float], best_direction: int):
        """Show debug visualization"""
        # Create visualization
        debug_img = screen.copy()
        
        # Draw clearness scores as bars
        height, width = debug_img.shape[:2]
        num_sectors = len(clearness_scores)
        sector_width = width // num_sectors
        
        for i, score in enumerate(clearness_scores):
            x = i * sector_width
            bar_height = int(score * 100)
            color = (0, 255, 0) if i == best_direction else (0, 165, 255)
            cv2.rectangle(debug_img, (x, height - bar_height), 
                         (x + sector_width - 2, height), color, -1)
        
        # Show obstacle mask
        obstacle_colored = cv2.cvtColor(obstacle_mask, cv2.COLOR_GRAY2BGR)
        obstacle_colored[:, :, 2] = obstacle_mask  # Red channel
        
        # Combine views
        combined = np.hstack([debug_img, obstacle_colored])
        
        # Resize for display
        display_height = 400
        aspect_ratio = combined.shape[1] / combined.shape[0]
        display_width = int(display_height * aspect_ratio)
        combined = cv2.resize(combined, (display_width, display_height))
        
        cv2.imshow('Navigator Debug', combined)
        cv2.waitKey(1)
    
    def _get_direction_name(self, direction_idx: int) -> str:
        """Get human-readable direction name"""
        names = ['FAR LEFT', 'FORWARD-LEFT', 'FORWARD', 'FORWARD-RIGHT', 
                'FAR RIGHT', 'BACK-RIGHT', 'BACK', 'BACK-LEFT']
        return names[direction_idx] if 0 <= direction_idx < len(names) else 'UNKNOWN'
    
    def stop(self):
        """Stop the navigator"""
        self.running = False
        self.movement_controller.stop()
        cv2.destroyAllWindows()
        print("Navigator stopped.")


def main():
    """Main entry point"""
    print("=" * 60)
    print("AUTONOMOUS GAME NAVIGATOR")
    print("=" * 60)
    print("\nThis bot will:")
    print("1. Capture your game screen")
    print("2. Detect obstacles automatically")
    print("3. Navigate autonomously while avoiding obstacles")
    print("\n⚠ IMPORTANT:")
    print("- Use for educational purposes only")
    print("- May violate game Terms of Service")
    print("- Switch to game window after countdown")
    print("=" * 60)
    
    # Load or create config
    try:
        config = BotConfig.from_file('config.json')
        print("\n✓ Loaded configuration from config.json")
    except:
        config = BotConfig()
        print("\n→ Using default configuration")
    
    # Create navigator
    navigator = AutonomousNavigator(config)
    
    # Start with debug mode (set to False to hide visualization)
    navigator.start(debug=True)


if __name__ == "__main__":
    main()
