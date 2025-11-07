"""
Evil Lands Configuration Tool
Interactively configure all detection regions for the RL farming agent
"""

import cv2
import numpy as np
import mss
import json
from pathlib import Path


class EvilLandsConfigurator:
    """Interactive configuration tool for Evil Lands"""
    
    def __init__(self):
        self.sct = mss.mss()
        self.config = {
            'game_region': [0, 0, 1920, 1080],
            'hp_bar_region': [10, 10, 200, 20],
            'xp_bar_region': [10, 35, 200, 20],
            'hp_text_region': [220, 10, 100, 20],
            'xp_text_region': [220, 35, 100, 20],
            'minimap_region': [1670, 50, 200, 200],
            'notification_region': [600, 100, 720, 200],
            'xp_bar_bottom_region': [400, 950, 1120, 30],
            'health_detection': {
                'detection_mode': 'region',
                'low_health_threshold': 30,
                'critical_health_threshold': 15
            },
            'enemy_detection': {
                'detection_method': 'hybrid',
                'min_hp_bar_area': 50,
                'max_hp_bar_area': 5000,
                'use_ocr': True
            },
            'reward_detection': {
                'use_ocr': True,
                'ocr_engine': 'easyocr'
            }
        }
        
        self.current_region_name = None
        self.current_region = None
        self.selecting = False
        self.start_point = None
        self.end_point = None
        self.screenshot = None
        self.display_image = None
        
        print("=" * 70)
        print(" Evil Lands Configuration Tool")
        print("=" * 70)
        print("\nThis tool will help you configure all detection regions.")
        print("Make sure Evil Lands is running in windowed mode!\n")
    
    def capture_screen(self):
        """Capture current screen"""
        monitor = self.sct.monitors[1]  # Primary monitor
        screenshot = self.sct.grab(monitor)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for region selection"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.selecting = True
            self.start_point = (x, y)
            self.end_point = (x, y)
        
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.selecting:
                self.end_point = (x, y)
        
        elif event == cv2.EVENT_LBUTTONUP:
            self.selecting = False
            self.end_point = (x, y)
            
            # Calculate region
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            x2 = max(self.start_point[0], self.end_point[0])
            y2 = max(self.start_point[1], self.end_point[1])
            
            width = x2 - x1
            height = y2 - y1
            
            if width > 5 and height > 5:  # Minimum size
                self.current_region = [x1, y1, width, height]
                print(f"  Selected: [{x1}, {y1}, {width}, {height}]")
    
    def draw_overlay(self):
        """Draw selection overlay"""
        self.display_image = self.screenshot.copy()
        
        # Draw existing regions
        self._draw_existing_regions()
        
        # Draw current selection
        if self.start_point and self.end_point:
            cv2.rectangle(
                self.display_image,
                self.start_point,
                self.end_point,
                (0, 255, 0),
                2
            )
            
            # Show dimensions
            x1 = min(self.start_point[0], self.end_point[0])
            y1 = min(self.start_point[1], self.end_point[1])
            width = abs(self.end_point[0] - self.start_point[0])
            height = abs(self.end_point[1] - self.start_point[1])
            
            text = f"{width}x{height}"
            cv2.putText(
                self.display_image,
                text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1
            )
    
    def _draw_existing_regions(self):
        """Draw all configured regions"""
        regions = {
            'hp_bar_region': ((255, 0, 0), 'HP Bar'),
            'xp_bar_region': ((0, 0, 255), 'XP Bar'),
            'hp_text_region': ((255, 100, 100), 'HP Text'),
            'xp_text_region': ((100, 100, 255), 'XP Text'),
            'minimap_region': ((0, 255, 255), 'Minimap'),
            'notification_region': ((255, 0, 255), 'Notifications'),
            'xp_bar_bottom_region': ((0, 255, 100), 'XP Bar Bottom'),
        }
        
        for region_name, (color, label) in regions.items():
            if region_name in self.config:
                region = self.config[region_name]
                x, y, w, h = region
                
                # Draw rectangle
                cv2.rectangle(
                    self.display_image,
                    (x, y),
                    (x + w, y + h),
                    color,
                    2
                )
                
                # Draw label
                cv2.putText(
                    self.display_image,
                    label,
                    (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )
    
    def configure_region(self, region_name, description, hints):
        """Configure a specific region"""
        print(f"\n{'='*70}")
        print(f"Configuring: {region_name}")
        print(f"Description: {description}")
        print(f"{'='*70}")
        
        for hint in hints:
            print(f"  • {hint}")
        
        print("\nInstructions:")
        print("  1. Click and drag to select the region")
        print("  2. Press ENTER to confirm")
        print("  3. Press 'r' to retry")
        print("  4. Press 's' to skip")
        print("  5. Press 'q' to quit")
        
        self.current_region_name = region_name
        self.current_region = None
        self.start_point = None
        self.end_point = None
        
        # Capture fresh screenshot
        self.screenshot = self.capture_screen()
        
        cv2.namedWindow('Configure Evil Lands', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Configure Evil Lands', 1280, 720)
        cv2.setMouseCallback('Configure Evil Lands', self.mouse_callback)
        
        while True:
            self.draw_overlay()
            cv2.imshow('Configure Evil Lands', self.display_image)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 13:  # Enter
                if self.current_region:
                    self.config[region_name] = self.current_region
                    print(f"✓ Region saved: {self.current_region}\n")
                    break
                else:
                    print("  No region selected. Try again or press 's' to skip.")
            
            elif key == ord('r'):  # Retry
                print("  Retrying...")
                self.start_point = None
                self.end_point = None
                self.current_region = None
            
            elif key == ord('s'):  # Skip
                print(f"  Skipped. Using default: {self.config.get(region_name, 'None')}\n")
                break
            
            elif key == ord('q'):  # Quit
                print("\n✗ Configuration cancelled.")
                cv2.destroyAllWindows()
                return False
        
        return True
    
    def configure_all(self):
        """Configure all regions interactively"""
        
        regions_to_configure = [
            # Player UI regions
            {
                'name': 'hp_bar_region',
                'description': 'Player HP Bar (Top Left, Red)',
                'hints': [
                    'Look at the TOP LEFT corner',
                    'Find the RED bar (your health)',
                    'Select the entire red bar',
                    'Should be horizontal and thin',
                ]
            },
            {
                'name': 'hp_text_region',
                'description': 'HP Numbers (e.g. "50/100")',
                'hints': [
                    'Look to the RIGHT of the HP bar',
                    'Find the numbers showing current/max HP',
                    'Select the text area only',
                    'Should show something like "50/100"',
                ]
            },
            {
                'name': 'xp_bar_region',
                'description': 'XP Bar (Below HP, Blue)',
                'hints': [
                    'Look BELOW the HP bar',
                    'Find the BLUE bar (your experience)',
                    'Select the entire blue bar',
                    'Should be similar size to HP bar',
                ]
            },
            {
                'name': 'xp_text_region',
                'description': 'XP Numbers (e.g. "1234/5000")',
                'hints': [
                    'Look to the RIGHT of the XP bar',
                    'Find the numbers showing current/max XP',
                    'Select the text area only',
                    'Should show something like "1234/5000"',
                ]
            },
            
            # Minimap
            {
                'name': 'minimap_region',
                'description': 'Minimap (Top Right, Circular)',
                'hints': [
                    'Look at the TOP RIGHT corner',
                    'Find the circular minimap',
                    'Select the entire minimap circle',
                    'Include some padding around it',
                ]
            },
            
            # Notifications and rewards
            {
                'name': 'notification_region',
                'description': 'Notification Area (Center Top)',
                'hints': [
                    'Look at the TOP CENTER of screen',
                    'This is where kill messages appear',
                    'Select a wide area where text shows up',
                    'Should cover "Enemy Defeated", XP gain, etc.',
                ]
            },
            
            # Bottom XP bar (if exists)
            {
                'name': 'xp_bar_bottom_region',
                'description': 'Bottom XP Bar (Optional)',
                'hints': [
                    'Some games have XP bar at BOTTOM',
                    'If Evil Lands has one, select it',
                    'Otherwise, press "s" to skip',
                    'Should be long horizontal bar',
                ]
            },
        ]
        
        print("\nStarting configuration process...")
        print("Make sure Evil Lands is visible on screen!\n")
        input("Press ENTER to begin...")
        
        for region_info in regions_to_configure:
            success = self.configure_region(
                region_info['name'],
                region_info['description'],
                region_info['hints']
            )
            
            if not success:
                return False
        
        cv2.destroyAllWindows()
        return True
    
    def test_regions(self):
        """Test all configured regions"""
        print("\n" + "="*70)
        print("Testing Configured Regions")
        print("="*70)
        
        # Capture screen
        screen = self.capture_screen()
        
        # Create visualization
        vis = screen.copy()
        
        # Draw all regions
        regions = {
            'hp_bar_region': ((255, 0, 0), 'HP Bar'),
            'xp_bar_region': ((0, 0, 255), 'XP Bar'),
            'hp_text_region': ((255, 100, 100), 'HP Text'),
            'xp_text_region': ((100, 100, 255), 'XP Text'),
            'minimap_region': ((0, 255, 255), 'Minimap'),
            'notification_region': ((255, 0, 255), 'Notifications'),
            'xp_bar_bottom_region': ((0, 255, 100), 'XP Bottom'),
        }
        
        print("\nConfigured regions:")
        for region_name, (color, label) in regions.items():
            if region_name in self.config:
                region = self.config[region_name]
                x, y, w, h = region
                
                print(f"  {label:20} [{x:4}, {y:4}, {w:4}, {h:4}]")
                
                # Draw on visualization
                cv2.rectangle(vis, (x, y), (x + w, y + h), color, 2)
                cv2.putText(vis, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Extract and show ROI
                roi = screen[y:y+h, x:x+w]
                roi_resized = cv2.resize(roi, (200, 50))
                
                # Show ROI in corner
                roi_y = 50 + len([k for k in self.config if k in regions and 
                                 list(regions.keys()).index(k) < list(regions.keys()).index(region_name)]) * 60
                vis[roi_y:roi_y+50, 10:210] = roi_resized
                cv2.rectangle(vis, (10, roi_y), (210, roi_y+50), color, 2)
        
        # Display
        cv2.namedWindow('Region Test', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Region Test', 1280, 720)
        cv2.imshow('Region Test', vis)
        
        print("\nReview the regions on screen.")
        print("Press any key to continue...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    def save_config(self, filename='config_rl.json'):
        """Save configuration to file"""
        filepath = Path(filename)
        
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        print(f"\n✓ Configuration saved to: {filepath.absolute()}")
        print(f"\nConfiguration summary:")
        print(json.dumps(self.config, indent=2))
    
    def run(self):
        """Run the configuration process"""
        try:
            # Configure all regions
            success = self.configure_all()
            
            if not success:
                print("\nConfiguration incomplete.")
                return
            
            # Test regions
            print("\nWould you like to test the regions?")
            response = input("Test regions? (y/n): ").strip().lower()
            
            if response == 'y':
                self.test_regions()
            
            # Save configuration
            print("\nWould you like to save the configuration?")
            response = input("Save config to config_rl.json? (y/n): ").strip().lower()
            
            if response == 'y':
                self.save_config()
                print("\n✓ Configuration complete!")
                print("\nNext steps:")
                print("  1. Test detection: python perception/health_detection.py")
                print("  2. Test enemies: python perception/enemy_detection.py")
                print("  3. Start training: python rl_farming_agent.py")
            else:
                print("\nConfiguration not saved.")
        
        except KeyboardInterrupt:
            print("\n\n✗ Configuration cancelled by user.")
        
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            cv2.destroyAllWindows()


def main():
    """Main entry point"""
    print("\n" + "="*70)
    print("  Evil Lands RL Agent - Configuration Tool")
    print("="*70)
    print("\nThis tool will help you configure all detection regions.")
    print("\nPreparation:")
    print("  1. Start Evil Lands")
    print("  2. Set to WINDOWED mode (not fullscreen)")
    print("  3. Position window so all UI is visible")
    print("  4. Make sure you can see:")
    print("     • HP bar (red, top left)")
    print("     • XP bar (blue, below HP)")
    print("     • HP/XP numbers (e.g. '50/100')")
    print("     • Minimap (top right)")
    print("\n")
    
    ready = input("Ready to start? (y/n): ").strip().lower()
    
    if ready != 'y':
        print("Configuration cancelled.")
        return
    
    configurator = EvilLandsConfigurator()
    configurator.run()


if __name__ == "__main__":
    main()
