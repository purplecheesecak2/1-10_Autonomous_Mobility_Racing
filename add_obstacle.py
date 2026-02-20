#!/usr/bin/env python3
"""
Add a static obstacle (parked car) to the Spielberg map PNG.
The obstacle is drawn as a filled black rectangle, which LiDAR detects as a wall.

Map parameters (from Spielberg_map.yaml):
  resolution: 0.05796 m/pixel
  origin: (-84.85359914210505, -36.30299725862132, 0.0)
  Image: 2000 x 2000 pixels

Usage:
  python3 add_obstacle.py                  # add obstacle at default position
  python3 add_obstacle.py --restore        # restore original map
  python3 add_obstacle.py --x 8.5 --y 2.0 # custom world position
"""

import argparse
import shutil
from pathlib import Path
from PIL import Image, ImageDraw

MAP_PNG  = Path('/home/louisdarong/sim_ws/src/f1tenth_gym_ros/maps/Spielberg_map.png')
BACKUP   = MAP_PNG.with_suffix('.png.backup')

# Map metadata
RESOLUTION = 0.05796   # m/pixel
ORIGIN_X   = -84.85359914210505
ORIGIN_Y   = -36.30299725862132
IMG_H      = 2000

# Obstacle size (meters) - approximate parked F1TENTH car footprint
OBS_LENGTH = 3.5   # m, along track (x direction here)
OBS_WIDTH  = 1.2   # m, perpendicular to track (y direction)


def world_to_pixel(wx: float, wy: float):
    """Convert world (m) to image (col, row) pixel coordinates."""
    col = int((wx - ORIGIN_X) / RESOLUTION)
    row = int(IMG_H - (wy - ORIGIN_Y) / RESOLUTION)
    return col, row


def add_obstacle(cx_world: float, cy_world: float,
                 length_m: float = OBS_LENGTH,
                 width_m:  float = OBS_WIDTH):
    """Draw a filled black rectangle at (cx_world, cy_world) in world frame."""

    # Backup original map once
    if not BACKUP.exists():
        shutil.copy(MAP_PNG, BACKUP)
        print(f"[backup] Original saved to {BACKUP}")

    img = Image.open(MAP_PNG).convert('RGB')
    draw = ImageDraw.Draw(img)

    # Obstacle extents in pixels
    half_len_px = int((length_m / 2) / RESOLUTION)
    half_wid_px = int((width_m  / 2) / RESOLUTION)

    cx_px, cy_px = world_to_pixel(cx_world, cy_world)

    x0 = cx_px - half_len_px
    x1 = cx_px + half_len_px
    y0 = cy_px - half_wid_px
    y1 = cy_px + half_wid_px

    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))

    img.save(MAP_PNG)
    print(f"[obstacle] Added at world ({cx_world:.2f}, {cy_world:.2f}) "
          f"→ pixel center ({cx_px}, {cy_px}), "
          f"rect=[({x0},{y0})→({x1},{y1})]")
    print(f"[obstacle] Size: {length_m}m x {width_m}m "
          f"= {x1-x0}px x {y1-y0}px")


def restore():
    """Restore original map from backup."""
    if BACKUP.exists():
        shutil.copy(BACKUP, MAP_PNG)
        print(f"[restore] Map restored from {BACKUP}")
    else:
        print("[restore] No backup found.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore', action='store_true',
                        help='Restore original map')
    parser.add_argument('--x', type=float, default=8.5,
                        help='Obstacle center x in world frame (m)')
    parser.add_argument('--y', type=float, default=2.0,
                        help='Obstacle center y in world frame (m)')
    parser.add_argument('--length', type=float, default=OBS_LENGTH,
                        help='Obstacle length along track (m)')
    parser.add_argument('--width', type=float, default=OBS_WIDTH,
                        help='Obstacle width across track (m)')
    args = parser.parse_args()

    if args.restore:
        restore()
    else:
        add_obstacle(args.x, args.y, args.length, args.width)
        print()
        print("Next: rebuild sim_ws and restart gym_bridge")
        print("  cd ~/sim_ws && colcon build && source install/setup.bash")
        print("  ros2 launch f1tenth_gym_ros gym_bridge_launch.py")
