"""
Test script to help configure screen region for the navigator
Run this to see your screen coordinates and adjust config.json
"""

import pyautogui
import time
from mss import mss
import cv2
import numpy as np

print("=" * 60)
print("SCREEN REGION CONFIGURATION HELPER")
print("=" * 60)
print("\nThis tool will help you find the correct screen region")
print("for your game window.\n")

# Get screen size
screen_width, screen_height = pyautogui.size()
print(f"Your screen resolution: {screen_width}x{screen_height}")

print("\n1. Move your mouse to the TOP-LEFT corner of the game window")
print("   Press Enter when ready...")
input()
top_left = pyautogui.position()
print(f"   Top-Left: ({top_left.x}, {top_left.y})")

print("\n2. Move your mouse to the BOTTOM-RIGHT corner of the game window")
print("   Press Enter when ready...")
input()
bottom_right = pyautogui.position()
print(f"   Bottom-Right: ({bottom_right.x}, {bottom_right.y})")

# Calculate region
left = top_left.x
top = top_left.y
width = bottom_right.x - top_left.x
height = bottom_right.y - top_left.y

print("\n" + "=" * 60)
print("YOUR SCREEN REGION CONFIGURATION:")
print("=" * 60)
print(f'"screen_region": [{left}, {top}, {width}, {height}]')
print("\nCopy this line into your config.json file!")
print("=" * 60)

# Test capture
print("\n3. Testing screen capture...")
print("   Capturing in 3 seconds - make sure game is visible...")
time.sleep(3)

sct = mss()
region = {"left": left, "top": top, "width": width, "height": height}
screenshot = sct.grab(region)
img = np.array(screenshot)
img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

# Save test image
cv2.imwrite('test_capture.png', img)
print(f"✓ Test capture saved as 'test_capture.png'")
print(f"  Size: {width}x{height}")

# Show preview
img_resized = cv2.resize(img, (800, 600))
cv2.imshow('Test Capture - Press any key to close', img_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n✓ Configuration complete!")
print("  Update config.json with the screen_region values above.")
