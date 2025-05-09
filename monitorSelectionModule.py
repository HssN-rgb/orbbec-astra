import autopy
from screeninfo import get_monitors
import cv2
import numpy as np


def get_all_monitors_info():
    monitors = get_monitors()
    for i, m in enumerate(monitors):
        print(f"Monitor #{i}: {m.name} - {m.width}x{m.height}, positioned at ({m.x}, {m.y})")
        if m.is_primary:
            print(f"  ↑ PRIMARY MONITOR")

    return monitors


def get_primary_monitor():
    monitors = get_monitors()

    # First try to find monitor marked as primary
    for m in monitors:
        if hasattr(m, 'is_primary') and m.is_primary:
            return m

    # If no monitor is marked as primary, return the first one (usually the primary)
    if monitors:
        return monitors[0]

    # Fallback dimensions if no monitors detected
    return type('obj', (object,), {'width': 1920, 'height': 1080, 'x': 0, 'y': 0})


def get_monitor_by_index(index=0):

    monitors = get_monitors()

    if 0 <= index < len(monitors):
        selected = monitors[index]
        return selected
    else:
        return get_primary_monitor()

#
# def move_mouse_on_monitor(x, y, monitor=None):
#     if monitor is None:
#         monitor = get_primary_monitor()
#
#     # Adjust coordinates to be relative to the entire screen space
#     absolute_x = monitor.x + x
#     absolute_y = monitor.y + y
#
#     # Ensure coordinates are within screen bounds
#     absolute_x = max(0, min(absolute_x, monitor.x + monitor.width - 1))
#     absolute_y = max(0, min(absolute_y, monitor.y + monitor.height - 1))
#
#     # Move the mouse
#     try:
#         autopy.mouse.move(absolute_x, absolute_y)
#         return True
#     except Exception as e:
#         print(f"Error moving mouse: {e}")
#         return False
