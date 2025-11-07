"""
Health and Mana Detection Module
Extracts HP and Mana percentages from game screen
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class HealthManaState:
    """Player health and XP state (Evil Lands specific)"""
    health_percentage: float  # 0-100
    health_current: int       # Current HP (e.g. 50)
    health_max: int           # Max HP (e.g. 100)
    xp_percentage: float      # 0-100 (XP bar, NOT mana)
    xp_current: int           # Current XP
    xp_max: int               # Max XP for level
    is_low_health: bool       # HP < 30%
    is_critical: bool         # HP < 15%
    detected: bool            # Successfully detected bars


class HealthDetector:
    """
    Detects health and mana bars from game screen
    
    Supports multiple detection methods:
    1. Color-based detection (red HP, blue mana)
    2. Fixed region parsing (if bars always in same place)
    3. Template matching (for bar backgrounds)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize health detector
        
        Args:
            config: Configuration dictionary with:
                - hp_bar_region: [x, y, width, height] - Red bar at top left
                - xp_bar_region: [x, y, width, height] - Blue bar below HP (NOT mana!)
                - hp_text_region: [x, y, width, height] - HP text (e.g. "50/100")
                - xp_text_region: [x, y, width, height] - XP text
                - low_health_threshold: 30 (default)
        """
        self.config = config or {}
        
        # Bar regions at top left (Evil Lands specific)
        # HP bar is red at very top left
        # XP bar is blue below HP bar
        self.hp_bar_region = self.config.get('hp_bar_region', [10, 10, 200, 20])
        self.xp_bar_region = self.config.get('xp_bar_region', [10, 35, 200, 20])
        
        # Text regions (numbers like "50/100")
        self.hp_text_region = self.config.get('hp_text_region', [220, 10, 100, 20])
        self.xp_text_region = self.config.get('xp_text_region', [220, 35, 100, 20])
        
        # Color ranges in HSV
        # Red HP bar: Hue 0-10 or 170-180 (red wraps around)
        self.hp_color_lower1 = np.array([0, 100, 100])    # Bright red
        self.hp_color_upper1 = np.array([10, 255, 255])
        self.hp_color_lower2 = np.array([170, 100, 100])  # Dark red
        self.hp_color_upper2 = np.array([180, 255, 255])
        
        # Blue XP bar: Hue 100-130
        self.xp_color_lower = np.array([100, 100, 100])
        self.xp_color_upper = np.array([130, 255, 255])
        
        # Thresholds
        self.low_health_threshold = self.config.get('low_health_threshold', 30)
        self.critical_health_threshold = self.config.get('critical_health_threshold', 15)
        
        # OCR setup for reading numbers
        self.use_ocr = self.config.get('use_ocr', True)
        self.ocr_reader = None
        if self.use_ocr:
            self._init_ocr()
        
        # Detection mode
        self.detection_mode = self.config.get('detection_mode', 'color')  # 'color', 'region', 'template'
        
        # Caching
        self.last_hp_region = None
        self.last_xp_region = None  # Changed from last_mana_region
        
        print(f"[HealthDetector] Initialized with mode: {self.detection_mode}")
        print(f"[HealthDetector] HP bar region: {self.hp_bar_region}")
        print(f"[HealthDetector] XP bar region: {self.xp_bar_region}")
    
    def _init_ocr(self):
        """Initialize OCR for reading HP/XP numbers"""
        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self.ocr_available = True
            print("[HealthDetector] OCR initialized for number reading")
        except:
            try:
                import easyocr
                self.ocr_reader = easyocr.Reader(['en'], gpu=False)
                self.ocr_available = True
                print("[HealthDetector] EasyOCR initialized")
            except:
                self.ocr_available = False
                print("[HealthDetector] OCR not available, using visual estimation only")
    
    def detect(self, screen: np.ndarray) -> HealthManaState:
        """
        Detect health and mana from screen
        
        Args:
            screen: Game screen image (BGR format)
            
        Returns:
            HealthManaState with detected values
        """
        if self.detection_mode == 'color':
            return self._detect_by_color(screen)
        elif self.detection_mode == 'region':
            return self._detect_by_region(screen)
        elif self.detection_mode == 'template':
            return self._detect_by_template(screen)
        else:
            # Fallback: try color first, then region
            result = self._detect_by_color(screen)
            if not result.detected:
                result = self._detect_by_region(screen)
            return result
    
    def _detect_by_color(self, screen: np.ndarray) -> HealthManaState:
        """
        Detect bars using color filtering
        
        Works by:
        1. Convert to HSV
        2. Create mask for red (HP) and blue (mana)
        3. Find contours
        4. Calculate fill percentage
        """
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(screen, cv2.COLOR_BGR2HSV)
            
            # Extract HP bar region (top left, red)
            hp_x, hp_y, hp_w, hp_h = self.hp_bar_region
            hp_roi = screen[hp_y:hp_y+hp_h, hp_x:hp_x+hp_w]
            hp_percentage = self._analyze_bar_fill(hp_roi, 'red')
            
            # Extract XP bar region (below HP, blue)
            xp_x, xp_y, xp_w, xp_h = self.xp_bar_region
            xp_roi = screen[xp_y:xp_y+xp_h, xp_x:xp_x+xp_w]
            xp_percentage = self._analyze_bar_fill(xp_roi, 'blue')
            
            # Try to read HP/XP numbers via OCR
            hp_current, hp_max = self._read_hp_numbers(screen)
            xp_current, xp_max = self._read_xp_numbers(screen)
            
            # If OCR got numbers, use them for accurate percentage
            if hp_current > 0 and hp_max > 0:
                hp_percentage = (hp_current / hp_max) * 100.0
            
            if xp_current > 0 and xp_max > 0:
                xp_percentage = (xp_current / xp_max) * 100.0
            
            # Check if detection was successful
            detected = hp_percentage is not None and xp_percentage is not None
            
            if not detected:
                return HealthManaState(
                    health_percentage=100.0,
                    health_current=0,
                    health_max=0,
                    xp_percentage=0.0,
                    xp_current=0,
                    xp_max=0,
                    is_low_health=False,
                    is_critical=False,
                    detected=False
                )
            
            # Calculate status flags
            is_low_health = hp_percentage < self.low_health_threshold
            is_critical = hp_percentage < self.critical_health_threshold
            
            return HealthManaState(
                health_percentage=hp_percentage or 100.0,
                health_current=hp_current,
                health_max=hp_max,
                xp_percentage=xp_percentage or 0.0,
                xp_current=xp_current,
                xp_max=xp_max,
                is_low_health=is_low_health,
                is_critical=is_critical,
                detected=True
            )
            
        except Exception as e:
            print(f"[HealthDetector] Color detection error: {e}")
            return HealthManaState(100.0, 0, 0, 0.0, 0, 0, False, False, False)
    
    def _analyze_bar_fill(self, bar_roi: np.ndarray, color: str) -> Optional[float]:
        """Analyze bar region for fill percentage"""
        if bar_roi.size == 0:
            return None
        
        # Convert to HSV
        hsv = cv2.cvtColor(bar_roi, cv2.COLOR_BGR2HSV)
        
        # Create mask based on color
        if color == 'red':
            mask1 = cv2.inRange(hsv, self.hp_color_lower1, self.hp_color_upper1)
            mask2 = cv2.inRange(hsv, self.hp_color_lower2, self.hp_color_upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:  # blue (XP bar)
            mask = cv2.inRange(hsv, self.xp_color_lower, self.xp_color_upper)
        
        return self._calculate_bar_percentage(mask, orientation='horizontal')
    
    def _read_hp_numbers(self, screen: np.ndarray) -> Tuple[int, int]:
        """Read HP numbers like '50/100' via OCR"""
        if not self.ocr_available:
            return 0, 0
        
        try:
            hp_text_x, hp_text_y, hp_text_w, hp_text_h = self.hp_text_region
            text_roi = screen[hp_text_y:hp_text_y+hp_text_h, hp_text_x:hp_text_x+hp_text_w]
            
            # Preprocess for OCR
            gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)
            # Threshold to white text on black
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # OCR
            text = self._run_ocr(binary)
            
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
    
    def _read_xp_numbers(self, screen: np.ndarray) -> Tuple[int, int]:
        """Read XP numbers like '1234/5000' via OCR"""
        if not self.ocr_available:
            return 0, 0
        
        try:
            xp_text_x, xp_text_y, xp_text_w, xp_text_h = self.xp_text_region
            text_roi = screen[xp_text_y:xp_text_y+xp_text_h, xp_text_x:xp_text_x+xp_text_w]
            
            # Preprocess
            gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
            
            # OCR
            text = self._run_ocr(binary)
            
            # Parse "1234/5000" format
            import re
            match = re.search(r'(\d+)\s*/\s*(\d+)', text)
            if match:
                current = int(match.group(1))
                max_xp = int(match.group(2))
                return current, max_xp
            
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
    
    def _find_red_bar_percentage(self, hsv: np.ndarray) -> Optional[float]:
        """Find HP bar and calculate percentage"""
        try:
            # Create masks for red (wraps around in HSV)
            mask1 = cv2.inRange(hsv, self.hp_color_lower1, self.hp_color_upper1)
            mask2 = cv2.inRange(hsv, self.hp_color_lower2, self.hp_color_upper2)
            hp_mask = cv2.bitwise_or(mask1, mask2)
            
            # If we have a known region, use it
            if self.hp_bar_region:
                x, y, w, h = self.hp_bar_region
                hp_mask = hp_mask[y:y+h, x:x+w]
                return self._calculate_bar_percentage(hp_mask, orientation='horizontal')
            
            # Otherwise, find largest red contour
            contours, _ = cv2.findContours(hp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find largest contour (likely the HP bar)
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Validate it looks like a bar (wide and short)
            aspect_ratio = w / max(h, 1)
            if aspect_ratio < 2:  # Not horizontal enough
                return None
            
            # Cache region for next time
            self.last_hp_region = (x, y, w, h)
            
            # Calculate percentage
            roi = hp_mask[y:y+h, x:x+w]
            return self._calculate_bar_percentage(roi, orientation='horizontal')
            
        except Exception as e:
            print(f"[HealthDetector] HP detection error: {e}")
            return None
    
    def _find_blue_bar_percentage(self, hsv: np.ndarray) -> Optional[float]:
        """Find XP bar and calculate percentage (LEGACY - use detect() instead)"""
        try:
            # Create mask for blue (XP bar)
            xp_mask = cv2.inRange(hsv, self.xp_color_lower, self.xp_color_upper)
            
            # If we have a known region, use it
            if self.xp_bar_region:
                x, y, w, h = self.xp_bar_region
                xp_mask = xp_mask[y:y+h, x:x+w]
                return self._calculate_bar_percentage(xp_mask, orientation='horizontal')
            
            # Otherwise, find largest blue contour
            contours, _ = cv2.findContours(xp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return None
            
            # Find largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Validate aspect ratio
            aspect_ratio = w / max(h, 1)
            if aspect_ratio < 2:
                return None
            
            # Cache region
            self.last_xp_region = (x, y, w, h)
            
            # Calculate percentage
            roi = xp_mask[y:y+h, x:x+w]
            return self._calculate_bar_percentage(roi, orientation='horizontal')
            
        except Exception as e:
            print(f"[HealthDetector] Mana detection error: {e}")
            return None
    
    def _calculate_bar_percentage(self, bar_mask: np.ndarray, orientation: str = 'horizontal') -> float:
        """
        Calculate bar fill percentage
        
        Args:
            bar_mask: Binary mask of the bar
            orientation: 'horizontal' or 'vertical'
            
        Returns:
            Percentage (0-100)
        """
        if bar_mask.size == 0:
            return 0.0
        
        if orientation == 'horizontal':
            # Sum each column, find rightmost filled column
            col_sums = np.sum(bar_mask, axis=0)
            filled_cols = np.where(col_sums > 0)[0]
            
            if len(filled_cols) == 0:
                return 0.0
            
            rightmost = filled_cols[-1]
            total_width = bar_mask.shape[1]
            percentage = (rightmost + 1) / total_width * 100
            
        else:  # vertical
            # Sum each row, find bottommost filled row
            row_sums = np.sum(bar_mask, axis=1)
            filled_rows = np.where(row_sums > 0)[0]
            
            if len(filled_rows) == 0:
                return 0.0
            
            bottommost = filled_rows[-1]
            total_height = bar_mask.shape[0]
            percentage = (bottommost + 1) / total_height * 100
        
        return min(100.0, max(0.0, percentage))
    
    def _detect_by_region(self, screen: np.ndarray) -> HealthManaState:
        """
        Detect bars using fixed screen regions (no recursion)
        """
        try:
            # Extract HP bar region
            hp_x, hp_y, hp_w, hp_h = self.hp_bar_region
            hp_roi = screen[hp_y:hp_y+hp_h, hp_x:hp_x+hp_w]
            hp_percentage = self._analyze_bar_fill(hp_roi, 'red')

            # Extract XP bar region
            xp_x, xp_y, xp_w, xp_h = self.xp_bar_region
            xp_roi = screen[xp_y:xp_y+xp_h, xp_x:xp_x+xp_w]
            xp_percentage = self._analyze_bar_fill(xp_roi, 'blue')

            # Try to read HP/XP numbers via OCR
            hp_current, hp_max = self._read_hp_numbers(screen)
            xp_current, xp_max = self._read_xp_numbers(screen)

            if hp_current > 0 and hp_max > 0:
                hp_percentage = (hp_current / hp_max) * 100.0
            if xp_current > 0 and xp_max > 0:
                xp_percentage = (xp_current / xp_max) * 100.0

            detected = hp_percentage is not None and xp_percentage is not None
            if not detected:
                return HealthManaState(
                    health_percentage=100.0,
                    health_current=0,
                    health_max=0,
                    xp_percentage=0.0,
                    xp_current=0,
                    xp_max=0,
                    is_low_health=False,
                    is_critical=False,
                    detected=False
                )

            is_low_health = hp_percentage < self.low_health_threshold
            is_critical = hp_percentage < self.critical_health_threshold

            return HealthManaState(
                health_percentage=hp_percentage or 100.0,
                health_current=hp_current,
                health_max=hp_max,
                xp_percentage=xp_percentage or 0.0,
                xp_current=xp_current,
                xp_max=xp_max,
                is_low_health=is_low_health,
                is_critical=is_critical,
                detected=True
            )
        except Exception as e:
            print(f"[HealthDetector] Region detection error: {e}")
            return HealthManaState(100.0, 0, 0, 0.0, 0, 0, False, False, False)
    
    def _analyze_bar_region(self, roi: np.ndarray, color: str) -> Optional[float]:
        """Analyze a bar region for fill percentage (LEGACY - redirects to _analyze_bar_fill)"""
        return self._analyze_bar_fill(roi, color)
    
    def _detect_by_template(self, screen: np.ndarray) -> HealthManaState:
        """
        Detect bars using template matching
        
        Requires template images of bar backgrounds
        TODO: Implement when we have templates
        """
        print("[HealthDetector] Template detection not yet implemented")
        # Fallback to region detection if template not implemented
        return self._detect_by_region(screen)
    
    def calibrate(self, screen: np.ndarray, manual_regions: Optional[Dict] = None):
        """
        Calibrate detector with current screen
        
        Args:
            screen: Game screen at full health
            manual_regions: Optional manual region specification
        """
        print("[HealthDetector] Calibrating...")
        
        if manual_regions:
            self.hp_bar_region = manual_regions.get('hp_bar_region')
            self.xp_bar_region = manual_regions.get('xp_bar_region')
            print(f"  HP region: {self.hp_bar_region}")
            print(f"  XP region: {self.xp_bar_region}")
        else:
            # Auto-detect bars
            result = self._detect_by_color(screen)
            if result.detected:
                print(f"  Auto-detected HP: {result.health_percentage:.1f}%")
                print(f"  Auto-detected XP: {result.xp_percentage:.1f}%")
            else:
                print("  Failed to auto-detect. Please provide manual regions.")
        
        print("[HealthDetector] Calibration complete")
    
    def visualize(self, screen: np.ndarray, state: HealthManaState) -> np.ndarray:
        """
        Draw health/XP detection on screen for debugging
        
        Args:
            screen: Original screen
            state: Detected state
            
        Returns:
            Screen with visualizations
        """
        vis = screen.copy()
        
        # Draw HP bar region
        if self.last_hp_region:
            x, y, w, h = self.last_hp_region
            cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(vis, f"HP: {state.health_percentage:.1f}% ({state.health_current}/{state.health_max})", 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw XP bar region
        if self.last_xp_region:
            x, y, w, h = self.last_xp_region
            cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(vis, f"XP: {state.xp_percentage:.1f}% ({state.xp_current}/{state.xp_max})", 
                       (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Draw status
        status_text = []
        if state.is_critical:
            status_text.append("CRITICAL HP!")
        elif state.is_low_health:
            status_text.append("Low HP")
        # XP doesn't have "low" warning, it just increases
        
        if status_text:
            cv2.putText(vis, " | ".join(status_text), (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        return vis


if __name__ == "__main__":
    """Test health detection"""
    import mss
    
    print("Health Detection Test")
    print("Press 'q' to quit, 'c' to calibrate")
    
    detector = HealthDetector()
    sct = mss.mss()
    
    # Monitor 1 (full screen)
    monitor = sct.monitors[1]
    
    while True:
        # Capture screen
        screenshot = sct.grab(monitor)
        screen = np.array(screenshot)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
        
        # Detect
        state = detector.detect(screen)
        
        # Visualize
        vis = detector.visualize(screen, state)
        
        # Resize for display
        vis = cv2.resize(vis, (1280, 720))
        
        cv2.imshow('Health Detection', vis)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('c'):
            detector.calibrate(screen)
    
    cv2.destroyAllWindows()
    print("Test complete")
