"""
Minimap-Based Navigation for Evil Lands
Uses the in-game minimap to follow paths and navigate intelligently
"""

import cv2
import numpy as np
import time
from mss import mss
import pyautogui
from dataclasses import dataclass
from typing import Tuple, List, Optional
import json
import math


@dataclass
class MinimapConfig:
    """Configuration for minimap-based navigation"""
    # Minimap location on screen (Evil Lands: top-right corner, largest circle, slightly down from top)
    # For BlueStacks 1920x1080: approximately [1670, 50, 200, 200]
    minimap_region: Tuple[int, int, int, int] = (1670, 50, 200, 200)  # (left, top, width, height)
    
    # Path detection colors (adjust based on minimap colors)
    path_color_lower: Tuple[int, int, int] = (100, 100, 100)  # Light areas = paths
    path_color_upper: Tuple[int, int, int] = (255, 255, 255)
    
    obstacle_color_lower: Tuple[int, int, int] = (0, 0, 0)  # Dark areas = obstacles
    obstacle_color_upper: Tuple[int, int, int] = (80, 80, 80)
    
    # Navigation settings
    movement_duration: float = 0.8  # Longer movement for smoother motion
    scan_interval: float = 0.2  # Faster scanning for responsive movement
    look_ahead_distance: int = 40  # Look further ahead
    turn_threshold: float = 20.0  # Less sensitive turning
    stuck_threshold: int = 10  # More frames before stuck detection
    continuous_movement: bool = True  # Keep moving instead of stop-start
    
    @classmethod
    def from_file(cls, filepath: str):
        """Load configuration from JSON file"""
        try:
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
                # Convert lists to tuples for color values
                if 'path_color_lower' in config_dict:
                    config_dict['path_color_lower'] = tuple(config_dict['path_color_lower'])
                if 'path_color_upper' in config_dict:
                    config_dict['path_color_upper'] = tuple(config_dict['path_color_upper'])
                if 'obstacle_color_lower' in config_dict:
                    config_dict['obstacle_color_lower'] = tuple(config_dict['obstacle_color_lower'])
                if 'obstacle_color_upper' in config_dict:
                    config_dict['obstacle_color_upper'] = tuple(config_dict['obstacle_color_upper'])
                if 'minimap_region' in config_dict:
                    config_dict['minimap_region'] = tuple(config_dict['minimap_region'])
                return cls(**config_dict)
        except FileNotFoundError:
            return cls()


class MinimapCapture:
    """Handles minimap screen capture and preprocessing"""
    
    def __init__(self, region: Tuple[int, int, int, int]):
        self.sct = mss()
        self.region = {
            "left": region[0],
            "top": region[1],
            "width": region[2],
            "height": region[3]
        }
        self.center = (region[2] // 2, region[3] // 2)  # Player position on minimap
    
    def capture(self) -> np.ndarray:
        """Capture minimap and return as numpy array"""
        screenshot = self.sct.grab(self.region)
        img = np.array(screenshot)
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img
    
    def preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess minimap for better path detection"""
        # Enhance contrast to compensate for transparency
        # Convert to LAB color space for better contrast adjustment
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to L channel
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        enhanced_lab = cv2.merge([l, a, b])
        img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
        
        # Increase overall brightness and contrast
        img = cv2.convertScaleAbs(img, alpha=1.3, beta=10)  # alpha=contrast, beta=brightness
        
        # Apply slight blur to reduce noise
        img = cv2.GaussianBlur(img, (3, 3), 0)
        return img


class MinimapPathFinder:
    """Analyzes minimap to find navigable paths"""
    
    def __init__(self, config: MinimapConfig):
        self.config = config
    
    def detect_paths(self, minimap: np.ndarray) -> np.ndarray:
        """
        Detect walkable paths on minimap
        Returns binary mask: 1 = path, 0 = obstacle
        """
        # Method 1: Color-based detection
        path_mask = cv2.inRange(minimap, 
                                np.array(self.config.path_color_lower),
                                np.array(self.config.path_color_upper))
        
        # Method 2: Brightness-based (lighter = walkable) - more aggressive
        gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
        _, bright_mask = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)  # Lower threshold for better detection
        
        # Method 3: Adaptive threshold for varying lighting
        adaptive_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                             cv2.THRESH_BINARY, 11, -2)
        
        # Combine all methods
        combined = cv2.bitwise_or(path_mask, bright_mask)
        combined = cv2.bitwise_or(combined, adaptive_mask)
        
        # Clean up with morphological operations - less aggressive to preserve paths
        kernel = np.ones((2, 2), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return combined
    
    def detect_obstacles(self, minimap: np.ndarray) -> np.ndarray:
        """
        Detect obstacles on minimap
        Returns binary mask: 1 = obstacle, 0 = clear
        """
        obstacle_mask = cv2.inRange(minimap,
                                    np.array(self.config.obstacle_color_lower),
                                    np.array(self.config.obstacle_color_upper))
        
        # Expand obstacles slightly for safety margin
        kernel = np.ones((5, 5), np.uint8)
        obstacle_mask = cv2.dilate(obstacle_mask, kernel, iterations=1)
        
        return obstacle_mask
    
    def find_path_direction(self, path_mask: np.ndarray, 
                           player_pos: Tuple[int, int],
                           look_ahead: int) -> Optional[float]:
        """
        Find best direction to move based on path analysis
        Returns angle in degrees (0 = up/north, 90 = right/east, etc.)
        """
        height, width = path_mask.shape
        center_x, center_y = player_pos
        
        # Analyze sectors around player (like a radar)
        num_rays = 16  # Check 16 directions
        best_angle = None
        best_score = -1
        
        for i in range(num_rays):
            angle = (360 / num_rays) * i
            angle_rad = math.radians(angle)
            
            # Cast ray in this direction
            ray_score = self._cast_ray(path_mask, center_x, center_y, 
                                      angle_rad, look_ahead)
            
            if ray_score > best_score:
                best_score = ray_score
                best_angle = angle
        
        return best_angle if best_score > 0.3 else None
    
    def _cast_ray(self, path_mask: np.ndarray, start_x: int, start_y: int,
                  angle: float, distance: int) -> float:
        """
        Cast a ray from player position and calculate path score
        Returns score 0-1 based on how clear the path is
        """
        points_on_path = 0
        total_points = 0
        
        # Sample points along the ray
        for d in range(5, distance, 3):
            x = int(start_x + d * math.sin(angle))
            y = int(start_y - d * math.cos(angle))  # Negative because y increases downward
            
            # Check if point is within bounds
            if 0 <= x < path_mask.shape[1] and 0 <= y < path_mask.shape[0]:
                total_points += 1
                if path_mask[y, x] > 0:
                    points_on_path += 1
            else:
                break
        
        return points_on_path / total_points if total_points > 0 else 0
    
    def follow_narrow_path(self, path_mask: np.ndarray, 
                           player_pos: Tuple[int, int]) -> Optional[float]:
        """
        Special algorithm to follow narrow paths (like trails)
        Uses path skeleton and direction tracking
        """
        # Create skeleton of path (thin lines)
        skeleton = self._skeletonize(path_mask)
        
        # Find path centerline ahead of player
        center_x, center_y = player_pos
        
        # Look ahead region (in front of player)
        look_ahead_region = skeleton[max(0, center_y - 40):center_y, :]
        
        if look_ahead_region.size == 0 or np.sum(look_ahead_region) < 10:
            return None
        
        # Find the main path direction using moments
        moments = cv2.moments(look_ahead_region)
        if moments['m00'] == 0:
            return None
        
        # Calculate centroid of path ahead
        cx = int(moments['m10'] / moments['m00'])
        cy = int(moments['m01'] / moments['m00']) + max(0, center_y - 40)
        
        # Calculate angle to path centroid
        dx = cx - center_x
        dy = center_y - cy  # Invert because y increases downward
        
        angle = math.degrees(math.atan2(dx, dy))
        if angle < 0:
            angle += 360
        
        return angle
    
    def _skeletonize(self, mask: np.ndarray) -> np.ndarray:
        """Create skeleton of path for narrow path following"""
        # Simple thinning to get path centerline
        skeleton = cv2.ximgproc.thinning(mask) if hasattr(cv2, 'ximgproc') else mask
        return skeleton
    
    def analyze_minimap(self, minimap: np.ndarray, player_pos: Tuple[int, int]) -> dict:
        """
        Comprehensive minimap analysis
        Returns dict with path info, obstacles, and recommended direction
        """
        # Detect paths and obstacles
        path_mask = self.detect_paths(minimap)
        obstacle_mask = self.detect_obstacles(minimap)
        
        # Mask out the character indicator at center (yellow arrow + V-shaped light cone)
        # The V-cone points upward (north) showing character's view direction
        center_x, center_y = player_pos
        
        # Create a triangular mask for the V-shaped cone pointing upward
        cone_length = 40  # How far the cone extends upward
        cone_width = 30   # Width of the cone at its widest point
        
        # Define triangle points: center bottom, top-left, top-right
        triangle_points = np.array([
            [center_x, center_y],                           # Bottom center (character position)
            [center_x - cone_width//2, center_y - cone_length],  # Top-left
            [center_x + cone_width//2, center_y - cone_length]   # Top-right
        ], np.int32)
        
        # Fill the triangle with black (0) to mask it out
        cv2.fillPoly(path_mask, [triangle_points], 0)
        cv2.fillPoly(obstacle_mask, [triangle_points], 0)
        
        # Also mask the circular center (the arrow itself)
        mask_radius = 15
        cv2.circle(path_mask, (center_x, center_y), mask_radius, 0, -1)
        cv2.circle(obstacle_mask, (center_x, center_y), mask_radius, 0, -1)
        
        # Try narrow path following first (more precise)
        narrow_path_angle = self.follow_narrow_path(path_mask, player_pos)
        
        # If no narrow path, use general direction finding
        if narrow_path_angle is None:
            general_angle = self.find_path_direction(
                path_mask, player_pos, self.config.look_ahead_distance
            )
        else:
            general_angle = narrow_path_angle
        
        # Check immediate surroundings for obstacles
        safety_check = self._check_immediate_area(obstacle_mask, player_pos)
        
        return {
            'recommended_angle': general_angle,
            'narrow_path': narrow_path_angle is not None,
            'path_mask': path_mask,
            'obstacle_mask': obstacle_mask,
            'is_safe': safety_check
        }
    
    def _check_immediate_area(self, obstacle_mask: np.ndarray, 
                             player_pos: Tuple[int, int]) -> bool:
        """Check if immediate area around player is safe"""
        cx, cy = player_pos
        radius = 10
        
        y_min = max(0, cy - radius)
        y_max = min(obstacle_mask.shape[0], cy + radius)
        x_min = max(0, cx - radius)
        x_max = min(obstacle_mask.shape[1], cx + radius)
        
        immediate_area = obstacle_mask[y_min:y_max, x_min:x_max]
        obstacle_count = np.sum(immediate_area)
        
        return obstacle_count < (immediate_area.size * 0.3)  # Less than 30% obstacles


class MinimapNavigator:
    """Main navigator using minimap-based pathfinding"""
    
    def __init__(self, config: MinimapConfig):
        self.config = config
        self.minimap_capture = MinimapCapture(config.minimap_region)
        self.path_finder = MinimapPathFinder(config)
        self.current_angle = 0  # Current facing direction
        self.last_position = None
        self.stuck_counter = 0
        self.running = False
    
    def angle_to_direction(self, angle: float) -> str:
        """Convert angle to arrow key direction"""
        # Normalize angle to 0-360
        angle = angle % 360
        
        # Map to 8 directions using arrow keys
        if 337.5 <= angle or angle < 22.5:
            return 'up'  # North
        elif 22.5 <= angle < 67.5:
            return 'up+right'  # Northeast
        elif 67.5 <= angle < 112.5:
            return 'right'  # East
        elif 112.5 <= angle < 157.5:
            return 'down+right'  # Southeast
        elif 157.5 <= angle < 202.5:
            return 'down'  # South
        elif 202.5 <= angle < 247.5:
            return 'down+left'  # Southwest
        elif 247.5 <= angle < 292.5:
            return 'left'  # West
        else:  # 292.5 <= angle < 337.5
            return 'up+left'  # Northwest
    
    def move_direction(self, direction: str):
        """Execute movement in specified direction using arrow keys"""
        # Parse direction string
        keys_to_press = []
        if '+' in direction:
            keys_to_press = direction.split('+')
        else:
            keys_to_press = [direction]
        
        # Press keys without releasing immediately for continuous movement
        for key in keys_to_press:
            pyautogui.keyDown(key)
        
        # Hold for movement duration
        time.sleep(self.config.movement_duration)
        
        # Only release if we're not doing continuous movement
        # Keys will be released before next direction or on stop
        if not self.config.continuous_movement:
            for key in keys_to_press:
                pyautogui.keyUp(key)
    
    def release_all_movement_keys(self):
        """Release all movement keys"""
        for key in ['up', 'down', 'left', 'right']:
            try:
                pyautogui.keyUp(key)
            except:
                pass
    
    def adjust_camera(self, target_angle: float):
        """Rotate camera to face target direction using numpad keys"""
        angle_diff = target_angle - self.current_angle
        
        # Normalize to -180 to 180
        while angle_diff > 180:
            angle_diff -= 360
        while angle_diff < -180:
            angle_diff += 360
        
        if abs(angle_diff) > self.config.turn_threshold:
            # Use numpad 4 (left) and 6 (right) for horizontal camera rotation
            if angle_diff > 0:
                # Turn right - use numpad 6
                turn_duration = abs(angle_diff) / 180  # Scale duration by angle
                pyautogui.keyDown('num6')
                time.sleep(min(turn_duration * 0.3, 0.5))  # Cap at 0.5 seconds
                pyautogui.keyUp('num6')
            else:
                # Turn left - use numpad 4
                turn_duration = abs(angle_diff) / 180
                pyautogui.keyDown('num4')
                time.sleep(min(turn_duration * 0.3, 0.5))
                pyautogui.keyUp('num4')
            
            self.current_angle = target_angle
    
    def start(self, debug: bool = True):
        """Start minimap-based navigation"""
        self.running = True
        
        print("=" * 60)
        print("MINIMAP-BASED NAVIGATOR")
        print("=" * 60)
        print("This bot uses your in-game minimap to navigate!")
        print("\n⚠ Make sure:")
        print("  1. Minimap is visible on screen")
        print("  2. config_minimap.json has correct minimap region")
        print("  3. Game is in focus after countdown")
        print("\nPress Ctrl+C to stop")
        print("\nStarting in 3 seconds...")
        time.sleep(3)
        
        try:
            self._navigation_loop(debug)
        except KeyboardInterrupt:
            print("\n\nStopping navigator...")
        finally:
            self.stop()
    
    def _navigation_loop(self, debug: bool):
        """Main navigation loop"""
        frame_count = 0
        last_direction = None
        
        while self.running:
            try:
                # Capture minimap
                minimap = self.minimap_capture.capture()
                minimap = self.minimap_capture.preprocess(minimap)
                
                # Analyze minimap
                analysis = self.path_finder.analyze_minimap(
                    minimap, self.minimap_capture.center
                )
                
                # Get recommended direction
                target_angle = analysis['recommended_angle']
                
                if target_angle is None:
                    print("⚠ No clear path detected - rotating to search...")
                    self.release_all_movement_keys()
                    pyautogui.keyDown('num6')
                    time.sleep(0.3)
                    pyautogui.keyUp('num6')
                    time.sleep(0.5)
                    last_direction = None
                    continue
                
                # Display info
                path_type = "NARROW PATH" if analysis['narrow_path'] else "OPEN AREA"
                safety = "✓ SAFE" if analysis['is_safe'] else "⚠ CAUTION"
                clearness = analysis.get('clearness', 0.5)
                print(f"{safety} | {path_type} | Target: {target_angle:.1f}° | Clearness: {clearness:.2f} | Frame: {frame_count}")
                
                # Adjust camera if needed (but not too often)
                if frame_count % 3 == 0:  # Only check camera every 3 frames
                    self.adjust_camera(target_angle)
                
                # Get movement direction
                direction = self.angle_to_direction(target_angle)
                
                # Only change keys if direction changed (for continuous movement)
                if direction != last_direction:
                    # Release old keys before pressing new ones
                    self.release_all_movement_keys()
                    time.sleep(0.05)  # Small delay for key release
                    
                # Move in recommended direction
                self.move_direction(direction)
                last_direction = direction
                
                # Debug visualization
                if debug:
                    self._show_debug(minimap, analysis, target_angle)
                
                # Check if stuck (but less aggressively)
                if frame_count % 5 == 0:  # Only check every 5 frames
                    self._check_stuck()
                
                # Wait before next iteration
                time.sleep(self.config.scan_interval)
                frame_count += 1
                
            except Exception as e:
                print(f"Error in navigation loop: {e}")
                self.release_all_movement_keys()
                time.sleep(1)
    
    def _show_debug(self, minimap: np.ndarray, analysis: dict, target_angle: float):
        """Show debug visualization"""
        debug_img = minimap.copy()
        height, width = debug_img.shape[:2]
        center = self.minimap_capture.center
        
        # Draw player position
        cv2.circle(debug_img, center, 5, (0, 255, 0), -1)
        
        # Draw target direction arrow
        if target_angle is not None:
            angle_rad = math.radians(target_angle)
            arrow_length = 30
            end_x = int(center[0] + arrow_length * math.sin(angle_rad))
            end_y = int(center[1] - arrow_length * math.cos(angle_rad))
            cv2.arrowedLine(debug_img, center, (end_x, end_y), 
                          (0, 255, 255), 2, tipLength=0.3)
        
        # Show path mask
        path_overlay = cv2.cvtColor(analysis['path_mask'], cv2.COLOR_GRAY2BGR)
        path_overlay[:, :, 1] = analysis['path_mask']  # Green channel
        
        # Show obstacle mask
        obstacle_overlay = cv2.cvtColor(analysis['obstacle_mask'], cv2.COLOR_GRAY2BGR)
        obstacle_overlay[:, :, 2] = analysis['obstacle_mask']  # Red channel
        
        # Combine views
        combined = np.hstack([debug_img, path_overlay, obstacle_overlay])
        
        # Resize for display
        scale = 2
        combined = cv2.resize(combined, (width * 3 * scale, height * scale), 
                            interpolation=cv2.INTER_NEAREST)
        
        # Add labels
        cv2.putText(combined, "Minimap", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(combined, "Paths", (width * scale + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Obstacles", (width * 2 * scale + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Debug window disabled to prevent blocking navigation
        # Uncomment these lines if you need to see the debug visualization:
        # cv2.imshow('Minimap Navigator Debug', combined)
        # cv2.waitKey(1)
    
    def _check_stuck(self):
        """Check if bot is stuck and take corrective action"""
        # This would need minimap position tracking
        # For now, simple counter-based approach
        self.stuck_counter += 1
        
        if self.stuck_counter > self.config.stuck_threshold:
            print("⚠ Might be stuck - trying to unstuck...")
            self.release_all_movement_keys()
            time.sleep(0.2)
            # Back up a bit
            pyautogui.keyDown('down')
            time.sleep(0.3)
            pyautogui.keyUp('down')
            # Rotate
            pyautogui.keyDown('num6')
            time.sleep(0.5)
            pyautogui.keyUp('num6')
            self.stuck_counter = 0
    
    def stop(self):
        """Stop the navigator"""
        self.running = False
        # Release all arrow keys and numpad keys
        self.release_all_movement_keys()
        for key in ['num4', 'num6', 'num8', 'num5']:
            try:
                pyautogui.keyUp(key)
            except:
                pass
        cv2.destroyAllWindows()
        print("Navigator stopped.")


def main():
    """Main entry point"""
    print("=" * 60)
    print("MINIMAP-BASED NAVIGATOR FOR EVIL LANDS")
    print("=" * 60)
    
    # Load config
    try:
        config = MinimapConfig.from_file('config_minimap.json')
        print("\n✓ Loaded configuration from config_minimap.json")
    except:
        config = MinimapConfig()
        print("\n→ Using default configuration")
        print("  Run 'python configure_minimap.py' to set up minimap region")
    
    # Create and start navigator
    navigator = MinimapNavigator(config)
    navigator.start(debug=True)


if __name__ == "__main__":
    main()
