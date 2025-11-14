"""
Reward Detection Module
Detects kills, loot, XP, level ups via OCR and visual cues
"""

import cv2
import numpy as np
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from collections import deque
import time


@dataclass
class RewardEvent:
    """Single reward event"""
    event_type: str  # 'kill', 'loot', 'xp', 'level_up', 'death', 'damage_taken', 'damage_dealt'
    value: float  # Numerical value (XP points, damage, etc.)
    timestamp: float
    confidence: float  # 0-1, how confident we are


@dataclass
class RewardState:
    """Accumulated rewards over time"""
    recent_events: List[RewardEvent] = field(default_factory=list)
    total_kills: int = 0
    total_loot: int = 0
    total_xp: float = 0.0
    total_damage_dealt: float = 0.0
    total_damage_taken: float = 0.0
    deaths: int = 0
    level_ups: int = 0
    
    # Performance metrics
    kills_per_minute: float = 0.0
    xp_per_minute: float = 0.0
    
    session_start: float = field(default_factory=time.time)


class RewardDetector:
    """
    Detects reward events from game screen
    
    Detection methods:
    1. OCR for text notifications (kill messages, XP gain)
    2. Template matching for icons (loot sparkles, level up)
    3. Screen region monitoring (XP bar changes)
    4. Damage number detection (floating combat text)
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize reward detector
        
        Args:
            config: Configuration dictionary with:
                - use_ocr: True/False
                - ocr_engine: 'tesseract' or 'easyocr'
                - notification_region: [x, y, width, height]
                - damage_regions: List of regions for damage numbers
                - xp_bar_region: [x, y, width, height]
        """
        self.config = config or {}
        
        # OCR setup
        self.use_ocr = self.config.get('use_ocr', True)
        self.ocr_engine = self.config.get('ocr_engine', 'easyocr')
        self.ocr_reader = None
        
        if self.use_ocr:
            self._init_ocr()
        
        # Game region (from config)
        self.game_region = self.config.get('game_region', [0, 0, 1920, 1080])
        self.screen_width = self.game_region[2]
        self.screen_height = self.game_region[3]
        
        # Detection region optimization (notifications/damage typically in center/top)
        self.use_detection_region = self.config.get('use_detection_region', True)
        self.detection_region_percent = self.config.get('detection_region_percent', 0.8)  # Center 80%
        
        # Screen regions
        self.notification_region = self.config.get('notification_region', [600, 100, 720, 200])
        self.damage_regions = self.config.get('damage_regions', [
            [800, 300, 320, 200],  # Center screen damage numbers
        ])
        self.xp_bar_region = self.config.get('xp_bar_region', [400, 950, 1120, 30])
        
        # Event tracking
        self.event_history = deque(maxlen=100)  # Last 100 events
        self.reward_state = RewardState()
        
        # Kill detection patterns
        self.kill_keywords = ['defeated', 'slain', 'killed', 'victory', 'enemy down']
        self.death_keywords = ['you died', 'defeated', 'respawn', 'game over']
        self.loot_keywords = ['obtained', 'looted', 'acquired', 'found', 'picked up']
        self.xp_keywords = ['xp', 'exp', 'experience', '+']
        self.levelup_keywords = ['level up', 'leveled up', 'new level']
        
        # Visual detection
        self.last_xp_bar_fill = 0.0
        self.frame_count = 0
        
        print(f"[RewardDetector] Initialized (OCR: {self.use_ocr}, Engine: {self.ocr_engine})")
        print(f"[RewardDetector] Game region: {self.screen_width}x{self.screen_height}")
        print(f"[RewardDetector] Detection region: {'ON' if self.use_detection_region else 'OFF'} ({self.detection_region_percent*100:.0f}%)")
    
    def _init_ocr(self):
        """Initialize OCR engine"""
        try:
            if self.ocr_engine == 'easyocr':
                try:
                    import easyocr
                    self.ocr_reader = easyocr.Reader(['en'], gpu=True)
                    print("[RewardDetector] EasyOCR initialized")
                except ImportError:
                    print("[RewardDetector] EasyOCR not installed, falling back to tesseract")
                    self.ocr_engine = 'tesseract'
            
            if self.ocr_engine == 'tesseract':
                try:
                    import pytesseract
                    # Test if tesseract is available
                    pytesseract.get_tesseract_version()
                    print("[RewardDetector] Tesseract initialized")
                except:
                    print("[RewardDetector] Tesseract not available, OCR disabled")
                    self.use_ocr = False
                    
        except Exception as e:
            print(f"[RewardDetector] OCR initialization error: {e}")
            self.use_ocr = False
    
    def detect(self, screen: np.ndarray, prev_screen: Optional[np.ndarray] = None) -> List[RewardEvent]:
        """
        Detect reward events from screen
        
        Args:
            screen: Current game screen (BGR)
            prev_screen: Previous frame for motion/change detection
            
        Returns:
            List of newly detected reward events
        """
        self.frame_count += 1
        events = []
        
        # 1. Check notification area for text
        if self.use_ocr and self.frame_count % 5 == 0:  # OCR every 5 frames (expensive)
            text_events = self._detect_from_notifications(screen)
            events.extend(text_events)
        
        # 2. Detect damage numbers (always check)
        damage_events = self._detect_damage_numbers(screen, prev_screen)
        events.extend(damage_events)
        
        # 3. Monitor XP bar changes
        if self.frame_count % 30 == 0:  # Check every 30 frames
            xp_event = self._detect_xp_change(screen)
            if xp_event:
                events.append(xp_event)
        
        # 4. Visual kill detection (blood splatter, enemy disappearance)
        # TODO: Implement advanced visual detection
        
        # Update reward state
        self._update_state(events)
        
        # Add to history
        for event in events:
            self.event_history.append(event)
        
        return events
    
    def _detect_from_notifications(self, screen: np.ndarray) -> List[RewardEvent]:
        """Extract text from notification region and parse for rewards"""
        events = []
        
        try:
            # Extract notification region
            x, y, w, h = self.notification_region
            notif_roi = screen[y:y+h, x:x+w]
            
            # Preprocess for OCR
            gray = cv2.cvtColor(notif_roi, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast
            gray = cv2.equalizeHist(gray)
            
            # Threshold
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Run OCR
            text = self._run_ocr(binary)
            text_lower = text.lower()
            
            # Parse for different event types
            
            # Kill detection
            for keyword in self.kill_keywords:
                if keyword in text_lower:
                    events.append(RewardEvent(
                        event_type='kill',
                        value=1.0,
                        timestamp=time.time(),
                        confidence=0.8
                    ))
                    break
            
            # Death detection
            for keyword in self.death_keywords:
                if keyword in text_lower:
                    events.append(RewardEvent(
                        event_type='death',
                        value=-1.0,
                        timestamp=time.time(),
                        confidence=0.9
                    ))
                    break
            
            # Loot detection
            for keyword in self.loot_keywords:
                if keyword in text_lower:
                    # Try to extract item rarity
                    rarity_value = self._parse_loot_rarity(text_lower)
                    events.append(RewardEvent(
                        event_type='loot',
                        value=rarity_value,
                        timestamp=time.time(),
                        confidence=0.7
                    ))
                    break
            
            # XP detection
            if any(keyword in text_lower for keyword in self.xp_keywords):
                xp_value = self._parse_xp_value(text)
                if xp_value > 0:
                    events.append(RewardEvent(
                        event_type='xp',
                        value=xp_value,
                        timestamp=time.time(),
                        confidence=0.6
                    ))
            
            # Level up detection
            for keyword in self.levelup_keywords:
                if keyword in text_lower:
                    events.append(RewardEvent(
                        event_type='level_up',
                        value=1.0,
                        timestamp=time.time(),
                        confidence=0.9
                    ))
                    break
            
        except Exception as e:
            print(f"[RewardDetector] Notification detection error: {e}")
        
        return events
    
    def _run_ocr(self, image: np.ndarray) -> str:
        """Run OCR on image"""
        try:
            if self.ocr_engine == 'easyocr' and self.ocr_reader:
                results = self.ocr_reader.readtext(image, detail=0)
                return ' '.join(results)
            
            elif self.ocr_engine == 'tesseract':
                import pytesseract
                text = pytesseract.image_to_string(image, config='--psm 6')
                return text
            
            return ""
            
        except Exception as e:
            print(f"[RewardDetector] OCR error: {e}")
            return ""
    
    def _detect_damage_numbers(self, screen: np.ndarray, prev_screen: Optional[np.ndarray]) -> List[RewardEvent]:
        """Detect floating damage numbers"""
        events = []
        
        # Damage numbers are usually:
        # - White/yellow (damage dealt)
        # - Red (damage taken)
        # - Green (healing)
        # - Appear and move upward
        
        try:
            for region in self.damage_regions:
                x, y, w, h = region
                roi = screen[y:y+h, x:x+w]
                
                # Convert to HSV
                hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                
                # Detect bright white/yellow (damage dealt)
                dealt_mask = cv2.inRange(hsv, np.array([15, 0, 200]), np.array([35, 100, 255]))
                
                # Detect red (damage taken)
                taken_mask1 = cv2.inRange(hsv, np.array([0, 150, 150]), np.array([10, 255, 255]))
                taken_mask2 = cv2.inRange(hsv, np.array([170, 150, 150]), np.array([180, 255, 255]))
                taken_mask = cv2.bitwise_or(taken_mask1, taken_mask2)
                
                # Count pixels (rough estimate)
                dealt_pixels = np.sum(dealt_mask > 0)
                taken_pixels = np.sum(taken_mask > 0)
                
                # If significant change (new damage numbers)
                if dealt_pixels > 100:  # Threshold
                    events.append(RewardEvent(
                        event_type='damage_dealt',
                        value=1.0,  # TODO: OCR to get actual number
                        timestamp=time.time(),
                        confidence=0.5
                    ))
                
                if taken_pixels > 100:
                    events.append(RewardEvent(
                        event_type='damage_taken',
                        value=1.0,  # TODO: OCR to get actual number
                        timestamp=time.time(),
                        confidence=0.5
                    ))
            
        except Exception as e:
            print(f"[RewardDetector] Damage detection error: {e}")
        
        return events
    
    def _detect_xp_change(self, screen: np.ndarray) -> Optional[RewardEvent]:
        """Detect XP bar filling"""
        try:
            x, y, w, h = self.xp_bar_region
            xp_roi = screen[y:y+h, x:x+w]
            
            # Convert to HSV
            hsv = cv2.cvtColor(xp_roi, cv2.COLOR_BGR2HSV)
            
            # XP bars are usually yellow/gold
            xp_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([35, 255, 255]))
            
            # Calculate fill percentage
            col_sums = np.sum(xp_mask, axis=0)
            filled_cols = np.where(col_sums > 0)[0]
            
            if len(filled_cols) == 0:
                fill_percentage = 0.0
            else:
                rightmost = filled_cols[-1]
                fill_percentage = (rightmost + 1) / xp_roi.shape[1]
            
            # Check if increased since last frame
            if fill_percentage > self.last_xp_bar_fill:
                xp_gained = (fill_percentage - self.last_xp_bar_fill) * 100  # Rough estimate
                self.last_xp_bar_fill = fill_percentage
                
                return RewardEvent(
                    event_type='xp',
                    value=xp_gained,
                    timestamp=time.time(),
                    confidence=0.7
                )
            
            self.last_xp_bar_fill = fill_percentage
            
        except Exception as e:
            print(f"[RewardDetector] XP detection error: {e}")
        
        return None
    
    def _parse_loot_rarity(self, text: str) -> float:
        """Parse loot rarity from text"""
        # Common = 1, Uncommon = 2, Rare = 3, Epic = 4, Legendary = 5
        if 'legendary' in text:
            return 5.0
        elif 'epic' in text:
            return 4.0
        elif 'rare' in text:
            return 3.0
        elif 'uncommon' in text:
            return 2.0
        else:
            return 1.0  # Common
    
    def _parse_xp_value(self, text: str) -> float:
        """Extract XP value from text like '+50 XP'"""
        import re
        
        # Look for numbers near XP keywords
        numbers = re.findall(r'\+?(\d+)', text)
        
        if numbers:
            return float(numbers[0])
        
        return 0.0
    
    def _update_state(self, events: List[RewardEvent]):
        """Update cumulative reward state"""
        for event in events:
            if event.event_type == 'kill':
                self.reward_state.total_kills += 1
            elif event.event_type == 'loot':
                self.reward_state.total_loot += 1
            elif event.event_type == 'xp':
                self.reward_state.total_xp += event.value
            elif event.event_type == 'death':
                self.reward_state.deaths += 1
            elif event.event_type == 'level_up':
                self.reward_state.level_ups += 1
            elif event.event_type == 'damage_dealt':
                self.reward_state.total_damage_dealt += event.value
            elif event.event_type == 'damage_taken':
                self.reward_state.total_damage_taken += event.value
            
            self.reward_state.recent_events.append(event)
        
        # Calculate performance metrics
        elapsed_time = (time.time() - self.reward_state.session_start) / 60.0  # Minutes
        if elapsed_time > 0:
            self.reward_state.kills_per_minute = self.reward_state.total_kills / elapsed_time
            self.reward_state.xp_per_minute = self.reward_state.total_xp / elapsed_time
    
    def get_state(self) -> RewardState:
        """Get current reward state"""
        return self.reward_state
    
    def reset_state(self):
        """Reset reward state (new session)"""
        self.reward_state = RewardState()
        self.event_history.clear()
        print("[RewardDetector] State reset")
    
    def visualize(self, screen: np.ndarray, events: List[RewardEvent]) -> np.ndarray:
        """Draw reward detection visualization"""
        vis = screen.copy()
        
        # Draw notification region
        x, y, w, h = self.notification_region
        cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 0, 255), 2)
        
        # Draw XP bar region
        x, y, w, h = self.xp_bar_region
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 255), 2)
        
        # Draw recent events
        y_offset = 90
        for event in events[-5:]:  # Last 5 events
            text = f"{event.event_type}: {event.value:.1f} ({event.confidence:.2f})"
            cv2.putText(vis, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 0), 1)
            y_offset += 20
        
        # Draw statistics
        stats = [
            f"Kills: {self.reward_state.total_kills} ({self.reward_state.kills_per_minute:.1f}/min)",
            f"XP: {self.reward_state.total_xp:.0f} ({self.reward_state.xp_per_minute:.1f}/min)",
            f"Loot: {self.reward_state.total_loot}",
            f"Deaths: {self.reward_state.deaths}"
        ]
        
        y_offset = screen.shape[0] - 100
        for stat in stats:
            cv2.putText(vis, stat, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.6, (255, 255, 0), 2)
            y_offset += 25
        
        return vis


if __name__ == "__main__":
    """Test reward detection"""
    import mss
    
    print("Reward Detection Test")
    print("Press 'q' to quit, 'r' to reset")
    
    detector = RewardDetector()
    sct = mss.mss()
    monitor = sct.monitors[1]
    
    prev_screen = None
    
    while True:
        screenshot = sct.grab(monitor)
        screen = np.array(screenshot)
        screen = cv2.cvtColor(screen, cv2.COLOR_BGRA2BGR)
        
        # Detect
        events = detector.detect(screen, prev_screen)
        
        # Visualize
        vis = detector.visualize(screen, events)
        vis = cv2.resize(vis, (1280, 720))
        
        cv2.imshow('Reward Detection', vis)
        
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset_state()
        
        prev_screen = screen
    
    cv2.destroyAllWindows()
