"""
Simple tool to find the BlueStacks window position
Move your mouse to the top-left corner of BlueStacks, then bottom-right
"""

import pyautogui
import time

print("=" * 60)
print("FIND BLUESTACKS WINDOW POSITION")
print("=" * 60)
print("\nThis will help you find the screen_region for your BlueStacks window")
print("\nInstructions:")
print("1. Position your BlueStacks window where you want it")
print("2. In 3 seconds, move your mouse to the TOP-LEFT corner of BlueStacks")
print("   (inside the window, not the title bar)")

for i in range(3, 0, -1):
    print(f"   {i}...")
    time.sleep(1)

top_left = pyautogui.position()
print(f"\n✓ Top-left corner: {top_left}")

print("\n3. Now move your mouse to the BOTTOM-RIGHT corner of BlueStacks")
print("   (inside the window)")

for i in range(3, 0, -1):
    print(f"   {i}...")
    time.sleep(1)

bottom_right = pyautogui.position()
print(f"\n✓ Bottom-right corner: {bottom_right}")

# Calculate region
left = top_left[0]
top = top_left[1]
width = bottom_right[0] - top_left[0]
height = bottom_right[1] - top_left[1]

print("\n" + "=" * 60)
print("RESULT - Add this to your config_minimap_default.json:")
print("=" * 60)
print(f'"screen_region": [{left}, {top}, {width}, {height}],')
print("\n" + "=" * 60)
print(f"\nYour BlueStacks window is at:")
print(f"  Position: ({left}, {top})")
print(f"  Size: {width} x {height}")
print("\nPress any key to exit...")
input()
