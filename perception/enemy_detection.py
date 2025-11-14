"""
Enemy Detection Module
Detects enemies, HP bars, target indicators
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class Enemy:
    """Detected enemy information"""
    position: Tuple[int, int]  # (x, y) screen coordinates
    hp_current: int            # Current HP (read from number in bar)
    hp_max: int                # Max HP (read from number)
    hp_percentage: Optional[float]  # 0-100, calculated from numbers
    has_attack_icon: bool      # Red attack icon above enemy
    is_targeted: bool          # Whether this enemy is currently targeted
    distance_score: float      # 0-1, estimated distance (0=far, 1=close)
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)


@dataclass
class EnemyState:
    """Overall enemy detection state"""
    enemies: List[Enemy]
    enemy_count: int
    has_target: bool
    nearest_enemy: Optional[Enemy]
    avg_distance: float
    detected: bool


class EnemyDetector:
    """
    Detects enemies on screen (Evil Lands specific)
    
    Detection methods:
    1. HP bars (can be red, white, or transparent) with health numbers inside
    2. Red attack icon (appears when you press spacebar to attack)
    3. Minimap red dots
    
    Key mechanics:
    - Enemy HP bars have numbers in the middle (e.g. "50/100")
    - Red attack icon appears above enemy when attacking (spacebar)
    - Keep attacking (spam spacebar) until red icon disappears
    - Press 'b' to collect loot after kill
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize enemy detector
        
        Args:
            config: Configuration dictionary with:
                - detection_method: 'hp_bars', 'attack_icon', 'minimap', 'hybrid'
                - hp_bar_colors: ['red', 'white', 'transparent'] - bars can be any color
                - attack_icon_color: Red icon color range
                - minimap_region: [x, y, width, height]
                - min_hp_bar_area: Minimum pixels for valid HP bar
                - use_ocr: True to read HP numbers from bars
        """
        self.config = config or {}
        
        # Detection method
        self.detection_method = self.config.get('detection_method', 'hybrid')
        
        # Enemy HP bar colors (Evil Lands: red, white, or transparent/gray)
        # Red bars
        self.enemy_hp_red_lower1 = np.array([0, 50, 50])
        self.enemy_hp_red_upper1 = np.array([10, 255, 255])
        self.enemy_hp_red_lower2 = np.array([170, 50, 50])
        self.enemy_hp_red_upper2 = np.array([180, 255, 255])
        
        # White/gray bars
        self.enemy_hp_white_lower = np.array([0, 0, 150])
        self.enemy_hp_white_upper = np.array([180, 50, 255])
        
        # Transparent/gray bars (low saturation)
        self.enemy_hp_gray_lower = np.array([0, 0, 100])
        self.enemy_hp_gray_upper = np.array([180, 100, 200])
        
        # Red attack icon (appears when attacking with spacebar)
        self.attack_icon_lower = np.array([0, 150, 150])
        self.attack_icon_upper = np.array([10, 255, 255])
        
        # Minimap enemy dots (red)
        self.minimap_enemy_lower = np.array([0, 150, 150])
        self.minimap_enemy_upper = np.array([10, 255, 255])
        
        # Constraints
        self.min_hp_bar_area = self.config.get('min_hp_bar_area', 50)
        self.max_hp_bar_area = self.config.get('max_hp_bar_area', 5000)
        self.min_hp_bar_aspect = 2.0  # HP bars are horizontal
        
        # Game region (full screen dimensions from config)
        self.game_region = self.config.get('game_region', [0, 0, 1920, 1080])
        self.screen_width = self.game_region[2]
        self.screen_height = self.game_region[3]
        
        # Detection region optimization (scan center area, not entire screen)
        self.use_detection_region = self.config.get('use_detection_region', True)
        self.detection_region_percent = self.config.get('detection_region_percent', 0.7)  # Center 70% of screen
        
        # Minimap config
        self.minimap_region = self.config.get('minimap_region', [1670, 50, 200, 200])
        
        # Smart scanning: use minimap to decide if full scan needed
        self.minimap_prefilter = self.config.get('minimap_prefilter', True)
        
        # OCR for reading HP numbers
        self.use_ocr = self.config.get('use_ocr', True)
        self.ocr_reader = None
        if self.use_ocr:
            self._init_ocr()
        
        # Caching
        self.last_enemies = []
        self.frame_count = 0
        
        print(f"[EnemyDetector] Initialized with method: {self.detection_method}")
        print(f"[EnemyDetector] OCR enabled: {self.use_ocr}")
        print(f"[EnemyDetector] Game region: {self.screen_width}x{self.screen_height}")
        print(f"[EnemyDetector] Detection region: {self.detection_region_percent*100:.0f}% of screen (optimized)")
        print(f"[EnemyDetector] Minimap pre-filter: {self.minimap_prefilter}")
    
    def _init_ocr(self):
        """Initialize OCR for reading HP numbers in bars"""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.ocr_available = True
            print("[EnemyDetector] OCR initialized")
        except:
            try:
                import easyocr
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
                self.ocr_available = True
                print("[EnemyDetector] EasyOCR initialized")
            except:
                self.ocr_available = False
                print("[EnemyDetector] OCR not available")
    
    def detect(self, screen: np.ndarray, minimap: Optional[np.ndarray] = None) -> EnemyState:
        """
        Detect all enemies on screen
        
        Args:
            screen: Full game screen (BGR)
            minimap: Minimap image (optional, BGR)
            
        Returns:
            EnemyState with all detected enemies
        """
        self.frame_count += 1
        
        # Smart scanning: check minimap first to avoid unnecessary full scans
        if self.minimap_prefilter and self.detection_method == 'hybrid':
            mm = minimap if minimap is not None else self._extract_minimap(screen)
            mm_enemies = self._detect_by_minimap(mm)
            
            # If no enemies on minimap, skip expensive screen scanning
            if len(mm_enemies) == 0:
                return EnemyState(
                    enemies=[],
                    enemy_count=0,
                    has_target=False,
                    nearest_enemy=None,
                    avg_distance=0.0,
                    detected=False
                )
        
        # Get detection region (cropped screen for faster processing)
        detection_screen, offset = self._get_detection_region(screen)
        
        if self.detection_method == 'hp_bars':
            enemies = self._detect_by_hp_bars(detection_screen, offset)
        elif self.detection_method == 'attack_icon':
            enemies = self._detect_by_attack_icon(detection_screen, offset)
        elif self.detection_method == 'minimap':
            enemies = self._detect_by_minimap(minimap if minimap is not None else self._extract_minimap(screen))
        elif self.detection_method == 'hybrid':
            # Combine multiple methods
            hp_enemies = self._detect_by_hp_bars(detection_screen, offset)
            icon_enemies = self._detect_by_attack_icon(detection_screen, offset)
            mm_enemies = self._detect_by_minimap(minimap if minimap is not None else self._extract_minimap(screen))
            enemies = self._merge_detections(hp_enemies, icon_enemies, mm_enemies)
        else:
            enemies = []
        
        # Calculate stats
        enemy_count = len(enemies)
        has_target = any(e.has_attack_icon for e in enemies)  # Has target if attack icon visible
        nearest_enemy = self._find_nearest_enemy(enemies)
        avg_distance = np.mean([e.distance_score for e in enemies]) if enemies else 0.0
        
        # Cache for next frame
        self.last_enemies = enemies
        
        return EnemyState(
            enemies=enemies,
            enemy_count=enemy_count,
            has_target=has_target,
            nearest_enemy=nearest_enemy,
            avg_distance=avg_distance,
            detected=enemy_count > 0
        )
    
    def _get_detection_region(self, screen: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Get optimized detection region (center portion of screen)
        
        Args:
            screen: Full screen (already cropped to game_region by agent)
            
        Returns:
            (cropped_screen, (offset_x, offset_y))
        """
        if not self.use_detection_region:
            return screen, (0, 0)
        
        height, width = screen.shape[:2]
        
        # Calculate center region based on actual screen dimensions
        region_width = int(width * self.detection_region_percent)
        region_height = int(height * self.detection_region_percent)
        
        # Center the detection region
        offset_x = (width - region_width) // 2
        offset_y = (height - region_height) // 2
        
        # Ensure offsets are within bounds
        offset_x = max(0, min(offset_x, width - region_width))
        offset_y = max(0, min(offset_y, height - region_height))
        
        # Crop to center region
        cropped = screen[offset_y:offset_y+region_height, offset_x:offset_x+region_width]
        
        return cropped, (offset_x, offset_y)
    
    def _detect_by_hp_bars(self, screen: np.ndarray, offset: Tuple[int, int] = (0, 0)) -> List[Enemy]:
        """
        Detect enemies by finding HP bars (red, white, or transparent) with numbers
        
        Args:
            screen: Screen region to scan (may be cropped)
            offset: (x, y) offset to convert local coords to full screen coords
        
        Evil Lands HP bars can be:
        - Red colored
        - White/gray colored
        - Transparent/translucent
        But all have health numbers in the middle (e.g. "50/100")
        """
        try:
            offset_x, offset_y = offset
            # Convert to HSV
            hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
            
            # Create masks for different bar colors
            # Red bars
            mask_red1 = cv2.inRange(hsv, self.enemy_hp_red_lower1, self.enemy_hp_red_upper1)
            mask_red2 = cv2.inRange(hsv, self.enemy_hp_red_lower2, self.enemy_hp_red_upper2)
            mask_red = cv2.bitwise_or(mask_red1, mask_red2)
            
            # White/gray bars
            mask_white = cv2.inRange(hsv, self.enemy_hp_white_lower, self.enemy_hp_white_upper)
            
            # Transparent/gray bars
            mask_gray = cv2.inRange(hsv, self.enemy_hp_gray_lower, self.enemy_hp_gray_upper)
            
            # Combine all masks
            hp_mask = cv2.bitwise_or(mask_red, mask_white)
            hp_mask = cv2.bitwise_or(hp_mask, mask_gray)
            
            # Morphology to connect nearby pixels
            kernel = np.ones((3, 7), np.uint8)
            hp_mask = cv2.morphologyEx(hp_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(hp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            enemies = []
            screen_height, screen_width = screen.shape[:2]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Filter by area
                if area < self.min_hp_bar_area or area > self.max_hp_bar_area:
                    continue
                
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check aspect ratio (HP bars are horizontal)
                aspect_ratio = w / max(h, 1)
                if aspect_ratio < self.min_hp_bar_aspect:
                    continue
                
                # Try to read HP numbers from the bar
                hp_current, hp_max = self._read_hp_from_bar(screen, x, y, w, h)
                
                # Calculate HP percentage
                if hp_current > 0 and hp_max > 0:
                    hp_percentage = (hp_current / hp_max) * 100.0
                else:
                    hp_percentage = None
                
                # Enemy position is below HP bar (approximate)
                # Add offset to convert back to full screen coordinates
                enemy_x = x + w // 2 + offset_x
                enemy_y = y + h + 30 + offset_y  # Estimate enemy is 30px below bar
                
                # Estimate distance (enemies closer = lower on screen usually)
                distance_score = enemy_y / self.screen_height
                
                enemies.append(Enemy(
                    position=(enemy_x, enemy_y),
                    hp_current=hp_current,
                    hp_max=hp_max,
                    hp_percentage=hp_percentage,
                    has_attack_icon=False,  # Will be set by attack icon detection
                    is_targeted=False,
                    distance_score=distance_score,
                    bbox=(x + offset_x, y + offset_y, w, h)  # Adjust bbox to full screen coords
                ))
            
            return enemies
            
        except Exception as e:
            print(f"[EnemyDetector] HP bar detection error: {e}")
            return []
    
    def _read_hp_from_bar(self, screen: np.ndarray, x: int, y: int, w: int, h: int) -> Tuple[int, int]:
        """
        Read HP numbers from enemy HP bar (e.g. "50/100")
        
        Args:
            screen: Full screen image
            x, y, w, h: Bounding box of HP bar
            
        Returns:
            (current_hp, max_hp)
        """
        if not self.ocr_available:
            return 0, 0
        
        try:
            # Extract HP bar region
            bar_roi = screen[y:y+h, x:x+w]
            
            # Preprocess for OCR - make text stand out
            gray = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2GRAY)
            
            # Try multiple thresholds
            _, binary1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            _, binary2 = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Enlarge for better OCR
            binary1 = cv2.resize(binary1, (w*2, h*2))
            
            # Run OCR
            text = self._run_ocr(binary1)
            
            # Parse "50/100" format
            import re
            match = re.search(r'(\d+)\s*/\s*(\d+)', text)
            if match:
                current = int(match.group(1))
                max_hp = int(match.group(2))
                return current, max_hp
            
        except Exception as e:
            pass
        
        return 0, 0
    
    def _run_ocr(self, image: np.ndarray) -> str:
        """Run OCR on image"""
        try:
            if self.ocr_reader:  # EasyOCR
                results = self.ocr_reader.readtext(image, detail=0)
                return ' '.join(results)
            else:  # Tesseract
                import pytesseract
                text = pytesseract.image_to_string(image, config='--psm 7 digits/')
                return text
        except:
            return ""
    
    def _detect_by_attack_icon(self, screen: np.ndarray, offset: Tuple[int, int] = (0, 0)) -> List[Enemy]:
        """
        Detect enemies by red attack icon (appears when pressing spacebar)
        
        Args:
            screen: Screen region to scan (may be cropped)
            offset: (x, y) offset to convert local coords to full screen coords
        
        This is the most reliable way to find actively attacked enemies.
        The red icon appears above the enemy when you attack.
        """
        try:
            offset_x, offset_y = offset
            # Convert to HSV
            hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
            
            # Mask for red attack icon
            icon_mask = cv2.inRange(hsv, self.attack_icon_lower, self.attack_icon_upper)
            
            # Morphology
            kernel = np.ones((5, 5), np.uint8)
            icon_mask = cv2.morphologyEx(icon_mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(icon_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            enemies = []
            screen_height, screen_width = screen.shape[:2]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if area < 20 or area > 2000:  # Icon size constraint
                    continue
                
                # Get position
                x, y, w, h = cv2.boundingRect(contour)
                
                # Enemy is below the icon
                # Add offset to convert back to full screen coordinates
                enemy_x = x + w // 2 + offset_x
                enemy_y = y + h + 20 + offset_y  # Icon is above enemy
                
                # Distance estimate (use configured screen height for consistency)
                distance_score = enemy_y / self.screen_height
                
                enemies.append(Enemy(
                    position=(enemy_x, enemy_y),
                    hp_current=0,  # Unknown without HP bar
                    hp_max=0,
                    hp_percentage=None,
                    has_attack_icon=True,  # This enemy is being attacked!
                    is_targeted=True,
                    distance_score=distance_score,
                    bbox=(x + offset_x, y + offset_y, w, h)  # Adjust bbox to full screen coords
                ))
            
            return enemies
            
        except Exception as e:
            print(f"[EnemyDetector] Attack icon detection error: {e}")
            return []
    
    def _estimate_hp_from_bar(self, bar_mask: np.ndarray) -> float:
        """Estimate HP percentage from HP bar mask"""
        if bar_mask.size == 0:
            return 100.0
        
        # Count filled columns from left
        col_sums = np.sum(bar_mask, axis=0)
        filled_cols = np.where(col_sums > 0)[0]
        
        if len(filled_cols) == 0:
            return 0.0
        
        rightmost = filled_cols[-1]
        total_width = bar_mask.shape[1]
        percentage = (rightmost + 1) / total_width * 100
        
        return min(100.0, max(0.0, percentage))
    
    def _detect_by_minimap(self, minimap: np.ndarray) -> List[Enemy]:
        """Detect enemies on minimap (red dots)"""
        try:
            if minimap is None or minimap.size == 0:
                return []
            
            # Convert to HSV
            hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
            
            # Mask for red dots
            mask1 = cv2.inRange(hsv, self.minimap_enemy_lower, self.minimap_enemy_upper)
            mask2 = cv2.inRange(hsv, np.array([170, 150, 150]), np.array([180, 255, 255]))
            enemy_mask = cv2.bitwise_or(mask1, mask2)
            
            # Find contours
            contours, _ = cv2.findContours(enemy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            enemies = []
            minimap_center = (minimap.shape[1] // 2, minimap.shape[0] // 2)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 10 or area > 500:  # Filter noise
                    continue
                
                # Get centroid
                M = cv2.moments(contour)
                if M["m00"] == 0:
                    continue
                
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                # Calculate distance from center (player position)
                dx = cx - minimap_center[0]
                dy = cy - minimap_center[1]
                distance = np.sqrt(dx*dx + dy*dy)
                
                # Normalize distance (0=close, 1=far)
                max_distance = minimap.shape[0] / 2
                distance_score = 1.0 - min(distance / max_distance, 1.0)
                
                # Approximate 3D position (rough estimate)
                # This is very approximate - real position needs game data
                x, y, w, h = cv2.boundingRect(contour)
                
                enemies.append(Enemy(
                    position=(cx, cy),  # Minimap coordinates
                    hp_current=0,       # Can't detect from minimap
                    hp_max=0,           # Can't detect from minimap
                    hp_percentage=None,  # Can't detect from minimap
                    has_attack_icon=False,  # Can't detect from minimap
                    is_targeted=False,
                    distance_score=distance_score,
                    bbox=(x, y, w, h)
                ))
            
            return enemies
            
        except Exception as e:
            print(f"[EnemyDetector] Minimap detection error: {e}")
            return []
    
    def _extract_minimap(self, screen: np.ndarray) -> np.ndarray:
        """Extract minimap region from screen"""
        x, y, w, h = self.minimap_region
        return screen[y:y+h, x:x+w]
    
    def _detect_target_indicator(self, screen: np.ndarray) -> Optional[Tuple[int, int]]:
        """Detect yellow target indicator/circle"""
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
            
            # Mask for yellow/orange
            target_mask = cv2.inRange(hsv, self.target_lower, self.target_upper)
            
            # Find largest contour
            contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            largest = max(contours, key=cv2.contourArea)
            
            # Get centroid
            M = cv2.moments(largest)
            if M["m00"] == 0:
                return None
            
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            
            return (cx, cy)
            
        except Exception as e:
            print(f"[EnemyDetector] Target indicator error: {e}")
            return None
    
    def _mark_targeted_enemy(self, enemies: List[Enemy], target_pos: Tuple[int, int]):
        """Mark which enemy is targeted based on target indicator position"""
        if not enemies:
            return
        
        # Find enemy closest to target indicator
        min_dist = float('inf')
        closest_idx = -1
        
        for i, enemy in enumerate(enemies):
            dx = enemy.position[0] - target_pos[0]
            dy = enemy.position[1] - target_pos[1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Mark as targeted if reasonably close (within 100 pixels)
        if closest_idx >= 0 and min_dist < 100:
            enemies[closest_idx].is_targeted = True
    
    def _find_nearest_enemy(self, enemies: List[Enemy]) -> Optional[Enemy]:
        """Find enemy with highest distance score (closest)"""
        if not enemies:
            return None
        
        return max(enemies, key=lambda e: e.distance_score)
    
    def _merge_detections(self, hp_enemies: List[Enemy], icon_enemies: List[Enemy], 
                         mm_enemies: List[Enemy]) -> List[Enemy]:
        """
        Merge enemies detected from HP bars, attack icons, and minimap
        
        Priority:
        1. Attack icon detection (most reliable for active target)
        2. HP bar detection (gives HP info)
        3. Minimap detection (backup)
        """
        merged = []
        
        # Start with icon enemies (being attacked)
        for icon_enemy in icon_enemies:
            # Try to find matching HP bar enemy
            matched = False
            for hp_enemy in hp_enemies:
                dx = icon_enemy.position[0] - hp_enemy.position[0]
                dy = icon_enemy.position[1] - hp_enemy.position[1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < 80:  # Within 80 pixels = same enemy
                    # Merge data - use HP bar data with attack icon flag
                    hp_enemy.has_attack_icon = True
                    hp_enemy.is_targeted = True
                    merged.append(hp_enemy)
                    hp_enemies.remove(hp_enemy)
                    matched = True
                    break
            
            if not matched:
                # No HP bar found, use icon detection alone
                merged.append(icon_enemy)
        
        # Add remaining HP bar enemies (not being attacked)
        merged.extend(hp_enemies)
        
        # Add minimap enemies that aren't already detected
        for mm_enemy in mm_enemies:
            too_close = False
            for existing in merged:
                dx = mm_enemy.position[0] - existing.position[0]
                dy = mm_enemy.position[1] - existing.position[1]
                dist = np.sqrt(dx*dx + dy*dy)
                
                if dist < 50:
                    too_close = True
                    break
            
            if not too_close:
                merged.append(mm_enemy)
        
        return merged
    
    def visualize(self, screen: np.ndarray, state: EnemyState) -> np.ndarray:
        """
        Draw enemy detection on screen
        
        Args:
            screen: Original screen
            state: Detected enemy state
            
        Returns:
            Screen with visualizations
        """
        vis = screen.copy()
        
        for enemy in state.enemies:
            # Draw bounding box
            x, y, w, h = enemy.bbox
            color = (0, 255, 255) if enemy.is_targeted else (0, 0, 255)  # Yellow if targeted, red otherwise
            thickness = 3 if enemy.is_targeted else 2
            cv2.rectangle(vis, (x, y), (x+w, y+h), color, thickness)
            
            # Draw enemy position marker
            ex, ey = enemy.position
            cv2.circle(vis, (ex, ey), 10, color, -1)
            
            # Draw HP if available
            if enemy.hp_percentage is not None:
                hp_text = f"HP: {enemy.hp_percentage:.0f}%"
                cv2.putText(vis, hp_text, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.4, color, 1)
            
            # Draw distance score
            dist_text = f"Dist: {enemy.distance_score:.2f}"
            cv2.putText(vis, dist_text, (ex+15, ey), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, color, 1)
        
        # Draw summary
        summary = f"Enemies: {state.enemy_count} | Targeted: {state.has_target}"
        cv2.putText(vis, summary, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 255, 0), 2)
        
        return vis


if __name__ == "__main__":
    """Test enemy detection"""
    import mss
    
    print("Enemy Detection Test")
    print("Press 'q' to quit")
    
    detector = EnemyDetector()
    sct = mss.mss()
    
    # Monitor 1
    monitor = sct.monitors[1]
    
    while True:
        # Capture
        screenshot = sct.grab(monitor)
        screen = np.array(screenshot)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
        
        # Detect
        state = detector.detect(screen)
        
        # Visualize
        vis = detector.visualize(screen, state)
        vis = cv2.resize(vis, (1280, 720))
        
        cv2.imshow('Enemy Detection', vis)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()
