import numpy as np
import cv2
import pyautogui
import mediapipe as mp
from openni import openni2
import tkinter as tk
from PIL import ImageGrab, ImageTk, Image, ImageDraw
import threading
import time
import os
from datetime import datetime


class ScreenFieldControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Screen Field and Hand Control")

        # Screen field selection variables
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.selection_made = False
        self.capturing = False
        self.capture_thread = None

        # Camera square mapping variables
        self.square_points_2d = []  # 2D points in image space
        self.square_points_3d = []  # 3D points with depth information
        self.mapping_active = False
        self.camera_thread = None
        self.touch_threshold = 50  # Default depth threshold for touch detection (mm)
        self.homography_matrix = None  # For perspective transformation
        
        # Hand detection range variables
        self.min_hand_distance = 300  # Minimum distance for hand detection (mm)
        self.max_hand_distance = 1500  # Maximum distance for hand detection (mm)
        
        # Grid visualization variables
        self.grid_resolution = 5  # Number of cells in each direction (configurable)
        self.show_grid = True  # Toggle grid visibility
        self.show_depth = True  # Toggle depth display
        self.grid_points_3d = []  # Will store (x, y, z) for each grid intersection
        self.grid_points_2d = []  # Will store (x, y) for each grid intersection
        self.depth_display_mode = 0  # 0: all points, 1: alternate points, 2: minimal
        
        # Enhanced touch detection variables
        self.use_grid_depth = True  # Use grid points for depth comparison
        self.grid_search_radius = 30  # Search radius for finding nearest grid point (pixels)
        self.grid_depth_tolerance = 30  # Tolerance for depth comparison with grid (mm)
        self.grid_depth_threshold = 30  # Specific threshold for click triggering (mm)
        self.nearest_grid_point = None  # Store nearest grid point for visualization
        self.debug_mode = True  # Show debug information

        # Wall plane estimation variables
        self.wall_plane = None  # Will store (a, b, c, d) for plane equation ax + by + cz + d = 0
        self.calibration_points = []  # 3D points for wall plane calibration
        self.calibration_mode = False

        # Drawing variables
        self.drawing_mode = False  # Toggle for drawing mode
        self.drawing_color = (0, 0, 255)  # Red color by default
        self.drawing_thickness = 3  # Line thickness
        self.drawing_points = []  # List to store drawing points
        self.last_point = None  # Last drawn point
        self.drawing_canvas = None  # Canvas for drawing
        self.is_touching = False  # Track if finger is currently touching
        self.touch_cooldown = 0  # Cooldown to prevent multiple touches
        
        # Available drawing colors
        self.color_options = [
            ("Red", (0, 0, 255)),
            ("Green", (0, 255, 0)),
            ("Blue", (255, 0, 0)),
            ("Yellow", (0, 255, 255)),
            ("Black", (0, 0, 0)),
            ("White", (255, 255, 255))
        ]
        
        self.canvas = None
        self.rect = None

        # Create frames for better organization
        control_frame = tk.Frame(root)
        control_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        slider_frame = tk.Frame(root)
        slider_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # GUI buttons
        self.choose_screen_button = tk.Button(control_frame, text="Choose Screen Field", command=self.open_selection_window)
        self.choose_screen_button.pack(pady=5)

        self.start_capture_button = tk.Button(control_frame, text="Start Capture", command=self.start_capture, state=tk.DISABLED)
        self.start_capture_button.pack(pady=5)
        """
        self.calibrate_button = tk.Button(control_frame, text="Calibrate Wall Plane", command=self.start_wall_calibration, state=tk.DISABLED)
        self.calibrate_button.pack(pady=5)"""

        self.square_mapping_button = tk.Button(control_frame, text="3D Square Mapping", command=self.start_square_mapping)
        self.square_mapping_button.pack(pady=5)

        # Add threshold slider
        self.threshold_label = tk.Label(slider_frame, text="Proximity Threshold (mm):")
        self.threshold_label.pack()
        self.threshold_slider = tk.Scale(slider_frame, from_=10, to=300, orient=tk.HORIZONTAL, command=self.update_threshold)
        self.threshold_slider.set(self.touch_threshold)
        self.threshold_slider.pack(fill=tk.X)
        
        # Add grid depth tolerance slider
        self.grid_tolerance_label = tk.Label(slider_frame, text="Grid Depth Tolerance (mm):")
        self.grid_tolerance_label.pack()
        self.grid_tolerance_slider = tk.Scale(slider_frame, from_=5, to=100, orient=tk.HORIZONTAL, 
                                              command=self.update_grid_depth_tolerance)
        self.grid_tolerance_slider.set(self.grid_depth_tolerance)
        self.grid_tolerance_slider.pack(fill=tk.X)

        # Add specific grid depth threshold slider for click triggering
        self.grid_threshold_label = tk.Label(slider_frame, text="Will add value to finger depth to trigger a click when: fingerdepth >= ave depth of radius (mm):")
        self.grid_threshold_label.pack()
        self.grid_threshold_slider = tk.Scale(slider_frame, from_=5, to=100, orient=tk.HORIZONTAL, 
                                             command=self.update_grid_depth_threshold)
        self.grid_threshold_slider.set(self.grid_depth_threshold)
        self.grid_threshold_slider.pack(fill=tk.X)

        # Add hand detection range sliders
        self.min_distance_label = tk.Label(slider_frame, text="Min Hand Distance (mm):")
        self.min_distance_label.pack()
        self.min_distance_slider = tk.Scale(slider_frame, from_=100, to=1000, orient=tk.HORIZONTAL, command=self.update_min_hand_distance)
        self.min_distance_slider.set(self.min_hand_distance)
        self.min_distance_slider.pack(fill=tk.X)
        
        self.max_distance_label = tk.Label(slider_frame, text="Max Hand Distance (mm):")
        self.max_distance_label.pack()
        self.max_distance_slider = tk.Scale(slider_frame, from_=500, to=3000, orient=tk.HORIZONTAL, command=self.update_max_hand_distance)
        self.max_distance_slider.set(self.max_hand_distance)
        self.max_distance_slider.pack(fill=tk.X)

        # Add grid resolution slider
        self.grid_label = tk.Label(slider_frame, text="Grid Resolution:")
        self.grid_label.pack()
        self.grid_slider = tk.Scale(slider_frame, from_=2, to=10, orient=tk.HORIZONTAL, command=self.update_grid_resolution)
        self.grid_slider.set(self.grid_resolution)
        self.grid_slider.pack(fill=tk.X)

        self.depth_value_limiter = 2000  # Default max acceptable depth value (mm)
        self.value_limiter_label = tk.Label(slider_frame, text="adjust this shit para di mag click kahit malapit yung kamay sa cam (mm):")
        self.value_limiter_label.pack()
        self.value_limiter_slider = tk.Scale(slider_frame, from_=500, to=3000, orient=tk.HORIZONTAL, 
                                            command=self.update_depth_value_limiter)
        self.value_limiter_slider.set(self.depth_value_limiter)
        self.value_limiter_slider.pack(fill=tk.X)

        # Add control buttons
        button_frame = tk.Frame(root)
        button_frame.pack(fill=tk.BOTH, padx=10, pady=5)
        
        # Add grid toggle button
        self.grid_toggle_button = tk.Button(button_frame, text="Toggle Grid", command=self.toggle_grid)
        self.grid_toggle_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add depth display toggle button
        self.depth_toggle_button = tk.Button(button_frame, text="Toggle Depth Display", command=self.toggle_depth)
        self.depth_toggle_button.pack(side=tk.LEFT, padx=5, pady=5)

        # Add depth display mode button
        self.depth_mode_button = tk.Button(button_frame, text="Cycle Depth Mode", command=self.cycle_depth_mode)
        self.depth_mode_button.pack(side=tk.LEFT, padx=5, pady=5)
        
        # Toggle grid-based depth detection
        self.grid_depth_button = tk.Button(button_frame, text="Use Grid Depth: ON", command=self.toggle_grid_depth)
        self.grid_depth_button.pack(side=tk.LEFT, padx=5, pady=5)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_all, state=tk.DISABLED)
        self.stop_button.pack(pady=10)

        # Wall angle related variables
        self.wall_angle = 0  # Angle in degrees (0 = perpendicular to camera)
        self.normal_vector = None  # Normal vector of the wall plane
        self.use_normal_projection = True  # Use projection onto wall normal for clicks
        self.adaptive_thresholds = True  # Adjust thresholds based on wall angle
        self.angle_sensitivity_factor = 1.0  # Sensitivity multiplier for angled surfaces
        
        # Coordinate display settings
        self.show_coordinates = True  # Toggle coordinate display
        self.grid_spacing = 50  # Spacing between coordinate grid lines in pixels

    def update_grid_depth_threshold(self, value):
        """Update grid depth threshold for click triggering"""
        self.grid_depth_threshold = int(value)
        print(f"Click depth threshold set to {self.grid_depth_threshold} mm")

    def update_depth_value_limiter(self, value):
        """Update the depth value limiter"""
        self.depth_value_limiter = int(value)
        print(f"Depth value limiter set to {self.depth_value_limiter} mm")

    def toggle_grid_depth(self):
        """Toggle grid-based depth detection"""
        self.use_grid_depth = not self.use_grid_depth
        if self.use_grid_depth:
            self.grid_depth_button.config(text="Use Grid Depth: ON", bg="light green")
            print("Grid-based depth detection enabled")
        else:
            self.grid_depth_button.config(text="Use Grid Depth: OFF", bg="SystemButtonFace")
            print("Grid-based depth detection disabled")
    
    def toggle_debug_mode(self):
        """Toggle debug mode"""
        self.debug_mode = not self.debug_mode
        if self.debug_mode:
            self.debug_button.config(text="Debug: ON", bg="light green")
            print("Debug mode enabled")
        else:
            self.debug_button.config(text="Debug: OFF", bg="SystemButtonFace")
            print("Debug mode disabled")
    
    def update_grid_depth_tolerance(self, value):
        """Update grid depth tolerance"""
        self.grid_depth_tolerance = int(value)
        print(f"Grid depth tolerance set to {self.grid_depth_tolerance} mm")

    def update_threshold(self, value):
        self.touch_threshold = int(value)
        print(f"Touch threshold updated to {self.touch_threshold} mm")
    
    def update_min_hand_distance(self, value):
        self.min_hand_distance = int(value)
        # Ensure min doesn't exceed max
        if self.min_hand_distance >= self.max_hand_distance:
            self.min_hand_distance = self.max_hand_distance - 100
            self.min_distance_slider.set(self.min_hand_distance)
        print(f"Minimum hand distance updated to {self.min_hand_distance} mm")
    
    def update_max_hand_distance(self, value):
        self.max_hand_distance = int(value)
        # Ensure max is greater than min
        if self.max_hand_distance <= self.min_hand_distance:
            self.max_hand_distance = self.min_hand_distance + 100
            self.max_distance_slider.set(self.max_hand_distance)
        print(f"Maximum hand distance updated to {self.max_hand_distance} mm")

    def update_grid_resolution(self, value):
        self.grid_resolution = int(value)
        print(f"Grid resolution updated to {self.grid_resolution}x{self.grid_resolution}")
        
        # Recalculate grid points if we have a valid square
        if len(self.square_points_3d) == 4:
            self.calculate_grid_points()

    def toggle_grid(self):
        self.show_grid = not self.show_grid
        print(f"Grid display {'enabled' if self.show_grid else 'disabled'}")

    def toggle_depth(self):
        self.show_depth = not self.show_depth
        print(f"Depth display {'enabled' if self.show_depth else 'disabled'}")

    def cycle_depth_mode(self):
        self.depth_display_mode = (self.depth_display_mode + 1) % 3
        modes = ["All Points", "Alternate Points", "Minimal"]
        print(f"Depth display mode: {modes[self.depth_display_mode]}")

    def draw_coordinate_grid(self, image):
        """Draw coordinate grid on the image"""
        h, w = image.shape[:2]
        grid_img = image.copy()
        
        # Draw horizontal lines
        for y in range(0, h, self.grid_spacing):
            cv2.line(grid_img, (0, y), (w-1, y), (200, 200, 200, 128), 1)
            cv2.putText(grid_img, f"{y}", (5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.4, (255, 0, 0), 1)
            
        # Draw vertical lines
        for x in range(0, w, self.grid_spacing):
            cv2.line(grid_img, (x, 0), (x, h-1), (200, 200, 200, 128), 1)
            cv2.putText(grid_img, f"{x}", (x+2, 15), cv2.FONT_HERSHEY_SIMPLEX, 
                      0.4, (255, 0, 0), 1)
            
        return grid_img
        
    def track_screen_mouse(self, event):
        """Track mouse coordinates on the captured screen"""
        x, y = event.x, event.y
        if hasattr(self, 'screen_status_label'):
            drawing_status = "DRAWING ENABLED" if self.drawing_mode else "Drawing disabled"
            self.screen_status_label.config(text=f"Position: ({x}, {y}) | {drawing_status}")

    def open_selection_window(self):
        # Open a fullscreen transparent window for square selection
        self.selection_window = tk.Toplevel(self.root)
        self.selection_window.attributes("-fullscreen", True)
        self.selection_window.attributes("-alpha", 0.3)
        self.selection_window.configure(bg='black')

        self.canvas = tk.Canvas(self.selection_window, bg='black', highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.selection_window.bind("<Button-1>", self.on_button_press)
        self.selection_window.bind("<B1-Motion>", self.on_mouse_drag)
        self.selection_window.bind("<ButtonRelease-1>", self.on_button_release)

    def on_button_press(self, event):
        # Record the starting point of the square
        self.start_x = event.x
        self.start_y = event.y
        if hasattr(self, 'rect') and self.rect:
            self.canvas.delete(self.rect)
        self.rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_mouse_drag(self, event):
        # Enforce square dimensions during drag
        side_length = max(abs(event.x - self.start_x), abs(event.y - self.start_y))
        self.canvas.coords(self.rect, self.start_x, self.start_y, self.start_x + side_length, self.start_y + side_length)

    def on_button_release(self, event):
        # Finalize the square selection and ensure dimensions are square
        side_length = max(abs(event.x - self.start_x), abs(event.y - self.start_y))
        self.end_x = self.start_x + side_length
        self.end_y = self.start_y + side_length
        self.selection_made = True
        self.selection_window.destroy()
        self.start_capture_button.config(state=tk.NORMAL)  # Enable Start Capture button
        self.calibrate_button.config(state=tk.NORMAL)  # Enable Calibrate button
        self.square_mapping_button.config(state=tk.NORMAL)  # Enable Square Mapping button
        
        # Initialize drawing canvas after selection
        self.initialize_drawing_canvas()

    def start_capture(self):
        if not self.selection_made or self.capturing:
            return

        self.capturing = True
        self.capture_thread = threading.Thread(target=self.capture_loop, daemon=True)
        self.capture_thread.start()
        self.stop_button.config(state=tk.NORMAL)  # Enable Stop button

    def capture_loop(self):
        try:
            while self.capturing:
                # Capture the selected square region
                bbox = (self.start_x, self.start_y, self.end_x, self.end_y)
                screen = ImageGrab.grab(bbox)
                screen_array = np.array(screen)
                
                # If drawing canvas exists, overlay it on the screen capture
                if hasattr(self, 'drawing_canvas') and self.drawing_canvas is not None:
                    # Overlay drawing on screen image
                    alpha_channel = self.drawing_canvas[:, :, 3] / 255.0
                    for c in range(3):  # RGB channels
                        screen_array[:, :, c] = screen_array[:, :, c] * (1 - alpha_channel) + \
                                              self.drawing_canvas[:, :, c] * alpha_channel
                
                # Draw coordinate grid if enabled
                if hasattr(self, 'show_coordinates') and self.show_coordinates:
                    screen_array = self.draw_coordinate_grid(screen_array)
                
                tk_image = ImageTk.PhotoImage(Image.fromarray(screen_array))

                # Display the captured screen dynamically
                if not hasattr(self, 'screen_window'):
                    self.screen_window = tk.Toplevel(self.root)
                    self.screen_window.title("Captured Screen with Drawing")
                    self.screen_label = tk.Label(self.screen_window, image=tk_image)
                    self.screen_label.image = tk_image
                    self.screen_label.pack()
                    
                    # Add status bar for coordinates and drawing info
                    self.screen_status_label = tk.Label(self.screen_window, 
                                                      text="Ready for drawing. Enable drawing mode to start.")
                    self.screen_status_label.pack(side=tk.BOTTOM, fill=tk.X)
                    
                    # Add mouse position tracking to captured screen window
                    self.screen_label.bind("<Motion>", self.track_screen_mouse)
                else:
                    self.screen_label.config(image=tk_image)
                    self.screen_label.image = tk_image

                time.sleep(0.05)  # Refresh at 20fps for smoother drawing
        except Exception as e:
            print(f"Error during screen capture: {e}")
    """
    def start_wall_calibration(self):
        #Start the wall plane calibration process
        self.calibration_mode = True
        self.calibration_points = []
        self.wall_plane = None
        print("Wall calibration mode activated")
        print("Please click on at least 4 points on the wall in the camera feed")
        
        # Start camera to collect calibration points
        self.camera_thread = threading.Thread(target=self.calibration_loop, daemon=True)
        self.camera_thread.start()"""

    def calibration_loop(self):
        """Camera loop for wall plane calibration"""
        try:
            # Initialize OpenNI2
            try:
                openni2.initialize(r'C:\Users\dell laptop\vscod\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs')
                print("OpenNI2 initialized successfully!")
            except Exception as e:
                print(f"Error initializing OpenNI2: {e}")
                return

            # Open device and streams
            try:
                dev = openni2.Device.open_any()
                depth_stream = dev.create_depth_stream()
                depth_stream.start()
                print("Depth stream started successfully!")

                color_stream = dev.create_color_stream()
                color_stream.start()
                print("Color stream started successfully!")
            except Exception as e:
                print(f"Failed to initialize streams: {e}")
                openni2.unload()
                return

            # OpenCV window for calibration
            cv2.namedWindow("Calibration")
            cv2.setMouseCallback("Calibration", self.calibration_click_handler)

            depth_array = None  # Store the latest depth array for use in mouse callback

            while self.calibration_mode:
                # Get color and depth frames
                color_frame = color_stream.read_frame()
                color_data = color_frame.get_buffer_as_uint8()
                color_image = np.frombuffer(color_data, dtype=np.uint8)
                color_image.shape = (color_frame.height, color_frame.width, 3)
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

                depth_frame = depth_stream.read_frame()
                depth_data = depth_frame.get_buffer_as_uint16()
                depth_array = np.frombuffer(depth_data, dtype=np.uint16)
                depth_array.shape = (depth_frame.height, depth_frame.width)
                
                # Store the latest depth_array for the callback
                self.latest_depth_array = depth_array
                
                # Draw calibration points
                for point in self.calibration_points:
                    cv2.circle(color_image, (int(point[0]), int(point[1])), 5, (0, 255, 0), -1)
                    # Display depth value
                    cv2.putText(color_image, f"{int(point[2])}mm", (int(point[0])+10, int(point[1])-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show instructions
                status_text = f"Click on wall: {len(self.calibration_points)}/4 points"
                cv2.putText(color_image, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (0, 255, 0), 2, cv2.LINE_AA)
                
                # If we have a plane, visualize it
                if self.wall_plane is not None:
                    cv2.putText(color_image, "Wall plane calibrated!", (20, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    plane_eq = f"Plane: {int(self.wall_plane[0])}x + {int(self.wall_plane[1])}y + {int(self.wall_plane[2])}z + {int(self.wall_plane[3])} = 0"
                    cv2.putText(color_image, plane_eq, (20, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                
                cv2.imshow("Calibration", color_image)
                
                # If we have enough points, compute the plane
                if len(self.calibration_points) >= 4 and self.wall_plane is None:
                    self.compute_wall_plane()
                    print(f"Wall plane computed: {self.wall_plane}")
                    # Keep calibration going to allow user to see the result
                    # but enable the square mapping button
                    self.square_mapping_button.config(state=tk.NORMAL)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.calibration_mode = False
                elif key == ord('r'):  # Reset calibration
                    self.calibration_points = []
                    self.wall_plane = None
                
            # Clean up
            depth_stream.stop()
            color_stream.stop()
            cv2.destroyWindow("Calibration")
            
        except Exception as e:
            print(f"Error in calibration loop: {e}")
        finally:
            openni2.unload()

    def calibration_click_handler(self, event, x, y, flags, param):
        """Handle mouse clicks during calibration"""
        if event == cv2.EVENT_LBUTTONDOWN and self.calibration_mode:
            if hasattr(self, 'latest_depth_array') and self.latest_depth_array is not None:
                # Get depth value from the click point
                if 0 <= y < self.latest_depth_array.shape[0] and 0 <= x < self.latest_depth_array.shape[1]:
                    depth_value = self.latest_depth_array[y, x]
                    if depth_value > 0:  # Ensure we have a valid depth value
                        # Store the 3D point (x, y, depth)
                        self.calibration_points.append((x, y, depth_value))
                        print(f"Added calibration point {len(self.calibration_points)}: ({x}, {y}, {int(depth_value)}mm)")
                    else:
                        print(f"Invalid depth value at ({x}, {y}). Please try another point.")
                else:
                    print(f"Point ({x}, {y}) is outside of depth array bounds.")

    def compute_wall_plane(self):
        """Compute the wall plane equation from calibration points using RANSAC with angle calculation"""
        if len(self.calibration_points) < 4:
            print("Not enough points to compute wall plane")
            return
        
        # Convert calibration points to numpy array
        points = np.array(self.calibration_points)
        
        # Use 3D points for plane fitting
        points_3d = np.zeros((len(points), 3))
        
        for i, (x, y, depth) in enumerate(points):
            # Convert from image coordinates to 3D world coordinates
            points_3d[i, 0] = x  # X coordinate (lateral)
            points_3d[i, 1] = y  # Y coordinate (vertical)
            points_3d[i, 2] = depth  # Z coordinate (depth)
        
        # Implement RANSAC for robust plane fitting
        best_plane = None
        best_inliers = 0
        iterations = 100
        threshold = 10  # mm distance threshold for inlier
        
        for _ in range(iterations):
            # Randomly select 3 points
            if len(points_3d) >= 3:
                idx = np.random.choice(len(points_3d), 3, replace=False)
                sample_points = points_3d[idx]
                
                # Calculate plane equation from these 3 points
                v1 = sample_points[1] - sample_points[0]
                v2 = sample_points[2] - sample_points[0]
                
                # Normal vector is the cross product
                normal = np.cross(v1, v2)
                
                # Skip if normal vector is zero (points are collinear)
                if np.linalg.norm(normal) < 1e-6:
                    continue
                    
                # Normalize the normal vector
                normal = normal / np.linalg.norm(normal)
                
                # Compute d in ax + by + cz + d = 0
                d = -np.dot(normal, sample_points[0])
                
                # Count inliers
                distances = np.abs(np.dot(points_3d, normal) + d)
                inliers = np.sum(distances < threshold)
                
                if inliers > best_inliers:
                    best_inliers = inliers
                    best_plane = (normal[0], normal[1], normal[2], d)
        
        if best_plane is not None:
            self.wall_plane = best_plane
            self.normal_vector = np.array(best_plane[:3])
            
            # Calculate wall angle relative to the camera's z-axis (depth axis)
            # Angle between normal vector and z-axis (0,0,1)
            z_axis = np.array([0, 0, 1])
            dot_product = np.dot(self.normal_vector, z_axis)
            # Ensure dot product is within valid range for arccos
            dot_product = max(-1.0, min(1.0, dot_product))
            angle_rad = np.arccos(dot_product)
            self.wall_angle = np.degrees(angle_rad)
            
            # Adjust sensitivity factor based on angle
            # As angle increases from 0 to 90, sensitivity increases
            if self.adaptive_thresholds:
                self.angle_sensitivity_factor = 1.0 + np.sin(angle_rad)
                
            print(f"Wall plane equation: {int(best_plane[0])}x + {int(best_plane[1])}y + {int(best_plane[2])}z + {int(best_plane[3])} = 0")
            print(f"Wall angle: {int(self.wall_angle)} degrees from perpendicular")
            print(f"Angle sensitivity factor: {int(self.angle_sensitivity_factor)}")
            print(f"Inliers: {best_inliers}/{len(points_3d)}")
        else:
            self.wall_plane = None
            self.normal_vector = None
            self.wall_angle = 0
            self.angle_sensitivity_factor = 1.0
            print("Failed to compute a reliable wall plane")
        
    def distance_to_wall(self, point_3d):
        """Calculate distance from a 3D point to the wall plane, optimized for angled walls"""
        if self.wall_plane is None:
            return float('inf')
        
        a, b, c, d = self.wall_plane
        numerator = abs(a*point_3d[0] + b*point_3d[1] + c*point_3d[2] + d)
        denominator = np.sqrt(a*a + b*b + c*c)
        
        if denominator < 1e-6:  # Avoid division by zero
            return float('inf')
        
        # Basic perpendicular distance to the plane
        perp_distance = numerator / denominator
        
        if self.use_normal_projection and self.normal_vector is not None:
            # For angled walls, scale based on the angle sensitivity factor
            if self.adaptive_thresholds and self.wall_angle > 15:
                perp_distance = perp_distance / self.angle_sensitivity_factor
        
        return perp_distance

    def start_square_mapping(self):
        if self.mapping_active:
            return

        self.mapping_active = True
        self.square_points_2d = []  # Reset 2D points
        self.square_points_3d = []  # Reset 3D points
        self.grid_points_2d = []    # Reset grid points
        self.grid_points_3d = []    # Reset 3D grid points
        self.nearest_grid_point = None  # Reset nearest grid point
        self.camera_thread = threading.Thread(target=self.camera_mapping_loop, daemon=True)
        self.camera_thread.start()
        self.stop_button.config(state=tk.NORMAL)  # Enable Stop button
        
        # Initialize the separate grid window
        self.initialize_grid_window()

    def find_nearest_grid_point(self, hand_pos_2d):
        """Find the nearest grid point to the hand position"""
        if not self.grid_points_2d or not self.grid_points_3d:
            return None
            
        hand_x, hand_y = hand_pos_2d
        
        nearest_point = None
        min_distance = float('inf')
        
        # Search through all grid points to find the nearest one
        for i, (grid_x, grid_y) in enumerate(self.grid_points_2d):
            # Calculate 2D Euclidean distance
            distance = np.sqrt((grid_x - hand_x)**2 + (grid_y - hand_y)**2)
            
            if distance < min_distance and distance < self.grid_search_radius:
                min_distance = distance
                nearest_point = self.grid_points_3d[i]
        
        return nearest_point

    def calculate_grid_points(self):
        """Calculate the 3D grid points inside the square"""
        if len(self.square_points_3d) != 4:
            return
            
        # Clear existing grid points
        self.grid_points_2d = []
        self.grid_points_3d = []
        
        # Extract the 4 corner points in 2D
        corners_2d = np.array(self.square_points_2d, dtype=np.float32)
        
        # Sort corners to ensure consistent order: top-left, top-right, bottom-right, bottom-left
        center = np.mean(corners_2d, axis=0)
        relative_positions = corners_2d - center
        angles = np.arctan2(relative_positions[:, 1], relative_positions[:, 0])
        sorted_indices = np.argsort(angles)
        corners_2d = corners_2d[sorted_indices]
        
        # Extract 3D points in the same order
        corners_3d = np.array([self.square_points_3d[i] for i in sorted_indices])
        
        # Generate grid points
        for i in range(self.grid_resolution + 1):
            for j in range(self.grid_resolution + 1):
                # Compute bilinear interpolation parameters
                u = i / self.grid_resolution
                v = j / self.grid_resolution
                
                # Bilinear interpolation
                p1 = (1-u)*(1-v)
                p2 = u*(1-v)
                p3 = u*v
                p4 = (1-u)*v
                
                # Compute interpolated 2D point
                x = p1*corners_2d[0][0] + p2*corners_2d[1][0] + p3*corners_2d[2][0] + p4*corners_2d[3][0]
                y = p1*corners_2d[0][1] + p2*corners_2d[1][1] + p3*corners_2d[2][1] + p4*corners_2d[3][1]
                
                # Compute interpolated depth
                z1 = corners_3d[0][2]
                z2 = corners_3d[1][2]
                z3 = corners_3d[2][2]
                z4 = corners_3d[3][2]
                z = p1*z1 + p2*z2 + p3*z3 + p4*z4
                
                # Store the grid point
                self.grid_points_2d.append((x, y))
                self.grid_points_3d.append((x, y, z))
                    
        print(f"Generated {len(self.grid_points_2d)} grid points")
        
        # Update the grid visualization window
        self.update_grid_window()
        
    def initialize_grid_window(self):
        """Create a separate window for grid visualization"""
        if hasattr(self, 'grid_window') and self.grid_window is not None:
            return
            
        self.grid_window = tk.Toplevel(self.root)
        self.grid_window.title("Grid Depth Visualization")
        self.grid_label = tk.Label(self.grid_window)
        self.grid_label.pack(padx=10, pady=10)
        
        # Add controls specific to the grid view
        control_frame = tk.Frame(self.grid_window)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Grid resolution control
        res_frame = tk.Frame(control_frame)
        res_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(res_frame, text="Grid Resolution:").pack(side=tk.LEFT)
        grid_res_slider = tk.Scale(res_frame, from_=2, to=10, orient=tk.HORIZONTAL, 
                                command=self.update_grid_resolution, length=100)
        grid_res_slider.set(self.grid_resolution)
        grid_res_slider.pack(side=tk.LEFT)
        
        # Depth display mode toggle
        tk.Button(control_frame, text="Toggle Depth", 
                command=self.toggle_depth).pack(side=tk.LEFT, padx=5)
        
        tk.Button(control_frame, text="Cycle Depth Mode", 
                command=self.cycle_depth_mode).pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.grid_status = tk.Label(self.grid_window, text="Grid visualization ready")
        self.grid_status.pack(side=tk.BOTTOM, fill=tk.X)

    def update_grid_window(self, depth_array=None):
        """Update the grid visualization window with current data"""
        if not hasattr(self, 'grid_window') or self.grid_window is None:
            self.initialize_grid_window()
            
        if len(self.square_points_3d) != 4:
            # Not enough points to draw grid yet
            grid_image = np.zeros((400, 400, 3), dtype=np.uint8)
            cv2.putText(grid_image, "Define 4 corner points first", 
                    (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            tk_image = ImageTk.PhotoImage(Image.fromarray(grid_image))
            self.grid_label.config(image=tk_image)
            self.grid_label.image = tk_image
            return
            
        # Create a clean image for grid visualization
        grid_size = 500  # Fixed size for grid window
        grid_image = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)
        
        if len(self.grid_points_2d) == 0:
            return
            
        # Get depth range for color mapping
        depths = [point[2] for point in self.grid_points_3d]
        min_depth = min(depths)
        max_depth = max(depths)
        
        # Scale grid points to fit the visualization window
        grid_points_array = np.array(self.grid_points_2d, dtype=np.float32)
        x_min, y_min = np.min(grid_points_array, axis=0)
        x_max, y_max = np.max(grid_points_array, axis=0)
        
        # Calculate scaling factors
        scale_x = (grid_size - 40) / (x_max - x_min)
        scale_y = (grid_size - 40) / (y_max - y_min)
        scale = min(scale_x, scale_y)  # Use the same scale for both dimensions to maintain aspect ratio
        
        # Scale and offset points to fit in the window with a margin
        scaled_grid_points = []
        for i, (x, y) in enumerate(self.grid_points_2d):
            scaled_x = int(20 + (x - x_min) * scale)
            scaled_y = int(20 + (y - y_min) * scale)
            scaled_grid_points.append((scaled_x, scaled_y))
        
        # Draw horizontal grid lines
        for i in range(self.grid_resolution + 1):
            row_points = scaled_grid_points[i*(self.grid_resolution+1):(i+1)*(self.grid_resolution+1)]
            row_depths = [self.grid_points_3d[i*(self.grid_resolution+1)+j][2] for j in range(self.grid_resolution+1)]
            
            for j in range(len(row_points)-1):
                pt1 = row_points[j]
                pt2 = row_points[j+1]
                
                # Color lines based on average depth
                avg_depth = (row_depths[j] + row_depths[j+1]) / 2
                line_color = self.get_depth_color(avg_depth, min_depth, max_depth)
                cv2.line(grid_image, pt1, pt2, line_color, 2)
        
        # Draw vertical grid lines
        for j in range(self.grid_resolution + 1):
            col_points = [scaled_grid_points[i*(self.grid_resolution+1)+j] for i in range(self.grid_resolution+1)]
            col_depths = [self.grid_points_3d[i*(self.grid_resolution+1)+j][2] for i in range(self.grid_resolution+1)]
            
            for i in range(len(col_points)-1):
                pt1 = col_points[i]
                pt2 = col_points[i+1]
                
                # Color lines based on average depth
                avg_depth = (col_depths[i] + col_depths[i+1]) / 2
                line_color = self.get_depth_color(avg_depth, min_depth, max_depth)
                cv2.line(grid_image, pt1, pt2, line_color, 2)
        
        # Draw depth values at grid intersections
        if self.show_depth:
            for i, ((x, y), (_, _, z)) in enumerate(zip(scaled_grid_points, self.grid_points_3d)):
                # Apply different display modes to control density of depth labels
                show_this_point = False
                
                if self.depth_display_mode == 0:
                    # All points
                    show_this_point = True
                elif self.depth_display_mode == 1:
                    # Alternate points (checkerboard pattern)
                    row = i // (self.grid_resolution + 1)
                    col = i % (self.grid_resolution + 1)
                    show_this_point = (row + col) % 2 == 0
                elif self.depth_display_mode == 2:
                    # Minimal points (just corners and center)
                    row = i // (self.grid_resolution + 1)
                    col = i % (self.grid_resolution + 1)
                    is_corner = (row == 0 and col == 0) or \
                                (row == 0 and col == self.grid_resolution) or \
                                (row == self.grid_resolution and col == 0) or \
                                (row == self.grid_resolution and col == self.grid_resolution)
                    is_center = row == self.grid_resolution // 2 and col == self.grid_resolution // 2
                    show_this_point = is_corner or is_center
                
                if show_this_point:
                    # Get color based on depth
                    depth_color = self.get_depth_color(z, min_depth, max_depth)
                    
                    # Draw point
                    cv2.circle(grid_image, (x, y), 5, depth_color, -1)
                    
                    # Create a darker background rectangle for text readability
                    text = f"{int(z)}"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(grid_image, 
                                (x-text_size[0]//2-2, y-text_size[1]//2-2), 
                                (x+text_size[0]//2+2, y+text_size[1]//2+2),
                                (0, 0, 0), -1)
                    
                    # Draw depth value centered on the point
                    cv2.putText(grid_image, text, (x-text_size[0]//2, y+text_size[1]//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add depth color legend
        if self.show_depth:
            legend_width = 300
            legend_height = 20
            legend_x = (grid_size - legend_width) // 2
            legend_y = grid_size - 50
            
            # Draw gradient bar
            for i in range(legend_width):
                normalized_depth = i / legend_width
                depth_value = min_depth + normalized_depth * (max_depth - min_depth)
                color = self.get_depth_color(depth_value, min_depth, max_depth)
                cv2.line(grid_image, (legend_x + i, legend_y), 
                    (legend_x + i, legend_y + legend_height), color, 1)
            
            # Draw min and max labels
            cv2.putText(grid_image, f"{int(min_depth)}mm", (legend_x - 5, legend_y + legend_height + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(grid_image, f"{int(max_depth)}mm", (legend_x + legend_width - 40, legend_y + legend_height + 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(grid_image, "Depth Values (mm)", (legend_x + legend_width//2 - 70, legend_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Add stats to status bar
        stats_text = f"Resolution: {self.grid_resolution}x{self.grid_resolution} | "
        stats_text += f"Depth range: {int(min_depth)}-{int(max_depth)}mm | "
        stats_text += f"Mode: {['All Points', 'Alternate Points', 'Minimal'][self.depth_display_mode]}"
        self.grid_status.config(text=stats_text)
        
        # Display the grid image
        tk_image = ImageTk.PhotoImage(Image.fromarray(grid_image))
        self.grid_label.config(image=tk_image)
        self.grid_label.image = tk_image

    def camera_mapping_loop(self):
        try:
            # Initialize OpenNI2 with the correct path
            try:
                openni2.initialize(r'C:\Users\dell laptop\vscod\OpenNI_2.3.0.86_202210111950_4c8f5aa4_beta6_windows\Win64-Release\sdk\libs')
                print("OpenNI2 initialized successfully!")
            except Exception as e:
                print(f"Error initializing OpenNI2: {e}")
                return

            # Open device and streams
            try:
                dev = openni2.Device.open_any()
                print("Device opened successfully!")

                depth_stream = dev.create_depth_stream()
                depth_stream.start()
                print("Depth stream started successfully!")

                color_stream = dev.create_color_stream()
                color_stream.start()
                print("Color stream started successfully!")
            except Exception as e:
                print(f"Failed to initialize streams: {e}")
                openni2.unload()
                return

            # Initialize MediaPipe for hand tracking
            mp_hands = mp.solutions.hands
            hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
            mp_drawing = mp.solutions.drawing_utils

            # OpenCV window for square mapping
            cv2.namedWindow("Camera Feed")
            cv2.setMouseCallback("Camera Feed", self.draw_3d_square)

            # Store the latest depth array for use in mouse callback
            self.latest_depth_array = None

            while self.mapping_active:
                # Capture frames
                color_frame = color_stream.read_frame()
                color_data = color_frame.get_buffer_as_uint8()
                color_image = np.frombuffer(color_data, dtype=np.uint8)
                color_image.shape = (color_frame.height, color_frame.width, 3)
                color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

                # Get depth frame
                depth_frame = depth_stream.read_frame()
                depth_data = depth_frame.get_buffer_as_uint16()
                depth_array = np.frombuffer(depth_data, dtype=np.uint16)
                depth_array.shape = (depth_frame.height, depth_frame.width)
                
                # Store the latest depth array for the callback
                self.latest_depth_array = depth_array

                # Draw the 3D square and grid on the camera feed
                color_image = self.draw_3d_square_on_image(color_image)
                """
                # Create a depth visualization for hand distance range
                depth_visualization = self.create_depth_range_visualization(depth_array)
                if depth_visualization is not None:
                    # Show the depth visualization window
                    cv2.imshow("Depth Range", depth_visualization)
                """
                # Hand tracking
                rgb_frame = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Index finger tip
                        index_finger_tip = hand_landmarks.landmark[8]
                        hand_x = int(index_finger_tip.x * color_frame.width)
                        hand_y = int(index_finger_tip.y * color_frame.height)

                        # Check if finger is within valid depth range
                        if 0 <= hand_x < depth_array.shape[1] and 0 <= hand_y < depth_array.shape[0]:
                            depth_value = depth_array[hand_y, hand_x]
                            print(f"Raw hand depth: {int(depth_value)}mm")
                            
                            # Only process hands within the configured distance range
                            if depth_value > 0 and self.min_hand_distance <= depth_value <= self.max_hand_distance:
                                # Draw hand landmarks
                                mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                                
                                # Create 3D point (x, y, z)
                                point_3d = (hand_x, hand_y, depth_value)
                                
                                # Calculate distance to wall
                                distance_to_wall = self.distance_to_wall(point_3d) if self.wall_plane else depth_value
                                print(f"Distance to wall: {int(distance_to_wall)}mm")
                                
                                # Check if finger is inside the projected 3D square
                                is_inside, mapped_pos = self.check_point_in_3d_square((hand_x, hand_y, depth_value))
                                
                                if is_inside:
                                    # Map finger position to screen coordinates
                                    screen_x, screen_y = mapped_pos
                                    
                                    # Move cursor
                                    pyautogui.moveTo(screen_x, screen_y)
                                    
                                    # UPDATED CLICK DETECTION: Use average depth around the finger
                                    should_click = False
                                    depth_value = depth_array[hand_y, hand_x]

                                    # Define a radius around the index finger to calculate average depth
                                    radius = 15  # pixels radius around the finger tip
                                    avg_depth = 0
                                    pixel_count = 0

                                    # Calculate average depth in the radius around the finger
                                    for dy in range(-radius, radius+1):
                                        for dx in range(-radius, radius+1):
                                            # Check if within radius (circular area)
                                            if dx*dx + dy*dy <= radius*radius:
                                                ny, nx = hand_y + dy, hand_x + dx
                                                # Check if within image bounds
                                                if 0 <= ny < depth_array.shape[0] and 0 <= nx < depth_array.shape[1]:
                                                    depth_at_point = depth_array[ny, nx]
                                                    if depth_at_point > 0:  # Valid depth value
                                                        avg_depth += depth_at_point
                                                        pixel_count += 1

                                    # Calculate average if we have valid pixels
                                    if pixel_count > 0:
                                        avg_depth = avg_depth / pixel_count
                                        
                                        # ENHANCED LOGIC: Apply the complete condition with value limiter
                                        # (depth_value + slider_value) >= average_depth_value < value_limiter_slider
                                        #should_click = ((depth_value + self.grid_depth_threshold) >= avg_depth) and (avg_depth < self.depth_value_limiter)
                                        should_click = ((depth_value + self.grid_depth_threshold) >= avg_depth) and (avg_depth < self.depth_value_limiter)
                                        # Debug information
                                        depth_diff = avg_depth - depth_value
                                        avg_depth_text = f"Finger:{int(depth_value)}mm Avg:{int(avg_depth)}mm Diff:{int(depth_diff)}mm"
                                        limiter_text = f"Limiter:{self.depth_value_limiter}mm"
                                        
                                        print(f"Finger depth: {int(depth_value)}mm, Average depth: {int(avg_depth)}mm, " +
                                            f"Diff: {int(depth_diff)}mm, Limiter: {self.depth_value_limiter}mm")
                                        
                                        # Show click threshold info
                                        if self.debug_mode:
                                            click_status = "CLICK" if should_click else "no click"
                                            threshold_text = f"Threshold: {self.grid_depth_threshold}mm ({click_status})"
                                            cv2.putText(color_image, threshold_text, 
                                                    (hand_x + 10, hand_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                                    0.5, (255, 255, 0), 1, cv2.LINE_AA)
                                            
                                            # Add limiter information
                                            cv2.putText(color_image, limiter_text, 
                                                    (hand_x + 10, hand_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 
                                                    0.5, (255, 255, 0), 1, cv2.LINE_AA)

                                        # Visual feedback for surroundings
                                        cv2.circle(color_image, (hand_x, hand_y), radius, (0, 255, 255), 1)  # Show sampling area

                                        # Draw circle around finger with color based on proximity
                                        proximity_indicator = int(min(255, max(0, 255 - (depth_diff / self.grid_depth_threshold) * 255)))
                                        cv2.circle(color_image, (hand_x, hand_y), 15, (0, proximity_indicator, 255-proximity_indicator), 2)
                                        
                                        # Show distance text
                                        cv2.putText(color_image, f"Depth diff: {int(depth_diff)}mm", 
                                                (hand_x + 10, hand_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                                0.5, (255, 255, 255), 1, cv2.LINE_AA)
                                        
                                        # Show average depth information if available
                                        if self.debug_mode:
                                            cv2.putText(color_image, avg_depth_text, 
                                                    (hand_x + 10, hand_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 
                                                    0.5, (255, 255, 0), 1, cv2.LINE_AA)
                                    # Handle touch
                                    if should_click:
                                        touch_pos = (int(screen_x-self.start_x), int(screen_y-self.start_y))
                                        
                                        # If in drawing mode, add point to drawing
                                        if self.drawing_mode:
                                            self.add_drawing_point(touch_pos)
                                            print(f"Drawing at {touch_pos}")
                                            
                                            # Visual feedback in camera feed for drawing
                                            cv2.circle(color_image, (hand_x, hand_y), 20, self.drawing_color, -1)
                                        else:
                                            # Standard click behavior when not in drawing mode
                                            print(f"Click at {touch_pos} - depth diff: {int(depth_diff)}mm")
                                            pyautogui.click(screen_x, screen_y)
                                            
                                            # Strong visual feedback for touch
                                            cv2.circle(color_image, (hand_x, hand_y), 20, (0, 0, 255), -1)
                                            
                                        # Update touch state - for drawing continuation
                                        self.is_touching = True
                                    else:
                                        # Reset touch state when finger moves away
                                        if self.is_touching:
                                            self.is_touching = False
                                            if self.drawing_mode:
                                                # End current line
                                                self.last_point = None
                                else:
                                    # Hand is inside detection range but outside mapped screen area
                                    cv2.putText(color_image, "Hand outside mapped area", 
                                              (hand_x + 10, hand_y - 10),
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

                # Show settings information
                cv2.putText(color_image, f"Click depth threshold: {self.grid_depth_threshold}mm", 
                           (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
                
                cv2.putText(color_image, f"Hand range: {self.min_hand_distance}-{self.max_hand_distance}mm", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                
                # Display drawing mode status
                if self.drawing_mode:
                    cv2.putText(color_image, "DRAWING MODE ON", 
                               (color_image.shape[1] - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                               0.7, self.drawing_color, 2, cv2.LINE_AA)
                
                if len(self.square_points_3d) < 4:
                    cv2.putText(color_image, f"Click to define 3D square: {len(self.square_points_3d)}/4 points", 
                               (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(color_image, "3D square mapped! Move finger inside square.", 
                               (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    cv2.putText(color_image, f"Grid resolution: {self.grid_resolution}x{self.grid_resolution}", 
                               (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                    
                    # Display depth mode
                    modes = ["All Points", "Alternate Points", "Minimal"]
                    cv2.putText(color_image, f"Depth mode: {modes[self.depth_display_mode]}", 
                               (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow("Camera Feed", color_image)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):  # Reset square points
                    self.square_points_2d = []
                    self.square_points_3d = []
                    self.grid_points_2d = []
                    self.grid_points_3d = []
                    self.homography_matrix = None
                elif key == ord('g'):  # Toggle grid display
                    self.toggle_grid()
                elif key == ord('d'):  # Toggle depth display
                    self.toggle_depth()
                elif key == ord('m'):  # Cycle depth display mode
                    self.cycle_depth_mode()
                elif key == ord('+') or key == ord('='):  # Increase grid resolution
                    self.grid_resolution = min(10, self.grid_resolution + 1)
                    self.grid_slider.set(self.grid_resolution)
                    self.calculate_grid_points()

                if len(self.grid_points_3d) == (self.grid_resolution + 1) * (self.grid_resolution + 1):
                    self.update_grid_window(depth_array)
                elif key == ord('-'):  # Decrease grid resolution
                    self.grid_resolution = max(2, self.grid_resolution - 1)
                    self.grid_slider.set(self.grid_resolution)
                    self.calculate_grid_points()
                elif key == ord('['):  # Decrease grid depth threshold
                    self.grid_depth_threshold = max(5, self.grid_depth_threshold - 5)
                    self.grid_threshold_slider.set(self.grid_depth_threshold)
                elif key == ord(']'):  # Increase grid depth threshold
                    self.grid_depth_threshold = min(100, self.grid_depth_threshold + 5)
                    self.grid_threshold_slider.set(self.grid_depth_threshold)
                elif key == ord('p'):  # Toggle drawing mode
                    self.toggle_drawing_mode()
                elif key == ord('c'):  # Clear drawing
                    self.clear_drawing()
                elif key == ord('s'):  # Save drawing
                    self.save_drawing()
                elif key == ord('t'):  # Toggle grid-based depth mode
                    self.toggle_grid_depth()
                elif key == ord('b'):  # Toggle debug mode
                    self.toggle_debug_mode()

            # Cleanup
            depth_stream.stop()
            color_stream.stop()
            openni2.unload()
            hands.close()
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Error during square mapping: {e}")
    """
    def create_depth_range_visualization(self, depth_array):
        #Create a visualization of the depth range for hand detection
        if depth_array is None:
            return None
            
        # Create a normalized depth visualization
        depth_vis = np.zeros_like(depth_array, dtype=np.uint8)
        
        # Only consider non-zero depth values
        mask = depth_array > 0
        if not np.any(mask):
            return None
            
        # Normalize depth values to 0-255 range for visualization
        min_val = np.min(depth_array[mask])
        max_val = np.max(depth_array[mask])
        
        # Avoid division by zero
        if max_val == min_val:
            return None
            
        # Create depth visualization
        depth_vis[mask] = 255 - 255 * (depth_array[mask] - min_val) / (max_val - min_val)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        
        # Draw detection range boundaries
        height, width = depth_array.shape
        bar_height = 20
        bar_image = np.zeros((bar_height, width, 3), dtype=np.uint8)
        
        # Map detection range to pixel positions
        min_pos = int(width * (self.min_hand_distance - min_val) / (max_val - min_val))
        max_pos = int(width * (self.max_hand_distance - min_val) / (max_val - min_val))
        
        # Ensure positions are within image bounds
        min_pos = max(0, min(width-1, min_pos))
        max_pos = max(0, min(width-1, max_pos))
        
        # Draw the detection range bar
        cv2.rectangle(bar_image, (0, 0), (width, bar_height), (50, 50, 50), -1)
        cv2.rectangle(bar_image, (min_pos, 0), (max_pos, bar_height), (0, 255, 0), -1)
        
        # Add text labels
        cv2.putText(bar_image, f"{int(min_val)}mm", (5, bar_height-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(bar_image, f"{int(max_val)}mm", (width-60, bar_height-5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Display detection range values
        cv2.putText(bar_image, f"Hand detection: {self.min_hand_distance}-{self.max_hand_distance}mm", 
                   (width//2-100, bar_height-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Concatenate the depth image and range bar
        vis_image = np.vstack((depth_vis, bar_image))
        
        return vis_image
    """
    def draw_3d_square(self, event, x, y, flags, param):
        """Handle clicks for defining 3D square corners"""
        if event == cv2.EVENT_LBUTTONDOWN and self.mapping_active:
            if len(self.square_points_3d) < 4:
                if hasattr(self, 'latest_depth_array') and self.latest_depth_array is not None:
                    # Get depth value from the click point
                    if 0 <= y < self.latest_depth_array.shape[0] and 0 <= x < self.latest_depth_array.shape[1]:
                        depth_value = self.latest_depth_array[y, x]
                        if depth_value > 0:  # Ensure we have a valid depth value
                            # Store both 2D and 3D points
                            self.square_points_2d.append((x, y))
                            self.square_points_3d.append((x, y, depth_value))
                            print(f"Added 3D square point {len(self.square_points_3d)}: ({x}, {y}, {int(depth_value)}mm)")
                            
                            # If we have 4 points, compute the homography matrix and grid points
                            if len(self.square_points_2d) == 4:
                                self.compute_perspective_transform()
                                self.calculate_grid_points()
                        else:
                            print(f"Invalid depth value at ({x}, {y}). Please try another point.")
                    else:
                        print(f"Point ({x}, {y}) is outside of depth array bounds.")

    def compute_perspective_transform(self):
        """Compute homography matrix for perspective transformation"""
        if len(self.square_points_2d) != 4:
            return
            
        # Source points are the 2D points in camera image
        src_pts = np.array(self.square_points_2d, dtype=np.float32)
        # Source points are the 2D points in camera image
        src_pts = np.array(self.square_points_2d, dtype=np.float32)
        
        # Destination points are the corners of the screen area we selected
        dst_pts = np.array([
            [self.start_x, self.start_y],
            [self.end_x, self.start_y],
            [self.end_x, self.end_y],
            [self.start_x, self.end_y]
        ], dtype=np.float32)
        
        # Compute homography matrix
        self.homography_matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        print("Perspective transform computed!")

    def check_point_in_3d_square(self, point_3d):
        """Check if a 3D point is inside the mapped 3D square and return mapped position"""
        if len(self.square_points_3d) != 4 or self.homography_matrix is None:
            return False, (0, 0)
            
        x, y, z = point_3d
        
        # First, check if the point is near the wall plane
        if self.wall_plane is not None:
            distance_to_wall = self.distance_to_wall(point_3d)
            if distance_to_wall > 100:  # Point is too far from wall plane (100mm threshold)
                return False, (0, 0)
        
        # Use perspective transform to map the 2D point to screen coordinates
        point_2d = np.array([[x, y]], dtype=np.float32)
        transformed_point = cv2.perspectiveTransform(point_2d.reshape(-1, 1, 2), self.homography_matrix)
        screen_x, screen_y = transformed_point[0][0]
        
        # Check if the mapped point is within the screen area
        if (self.start_x <= screen_x <= self.end_x and 
            self.start_y <= screen_y <= self.end_y):
            return True, (screen_x, screen_y)
        
        return False, (0, 0)

    def get_depth_color(self, depth_value, min_depth=700, max_depth=1500):
        """Get color based on depth value (green-yellow-red gradient)"""
        # Normalize depth to 0-1 range
        normalized = max(0, min(1, (depth_value - min_depth) / (max_depth - min_depth)))
        
        # Create color gradient: green (0) -> yellow (0.5) -> red (1)
        if normalized < 0.5:
            # Green to yellow
            r = int(255 * (2 * normalized))
            g = 255
            b = 0
        else:
            # Yellow to red
            r = 255
            g = int(255 * (2 - 2 * normalized))
            b = 0
            
        return (b, g, r)  # OpenCV uses BGR

    def draw_3d_square_on_image(self, image):
        """Draw the 3D square and grid on the image with wall angle visualization"""
        # Draw individual corner points
        for i, (x, y, z) in enumerate(self.square_points_3d):
            cv2.circle(image, (int(x), int(y)), 5, (0, 0, 255), -1)
            cv2.putText(image, f"P{i+1}: {int(z)}mm", (int(x)+10, int(y)-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        
        # Draw complete square if we have 4 points
        if len(self.square_points_2d) == 4:
            # Draw lines connecting the points
            pts = np.array(self.square_points_2d, np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
            
            # Draw grid if enabled
            if self.show_grid and len(self.grid_points_2d) > 0:
                # Draw grid lines
                grid_points = np.array(self.grid_points_2d, dtype=np.int32)
                
                # Get min and max depth for color mapping
                depths = [point[2] for point in self.grid_points_3d]
                min_depth = min(depths)
                max_depth = max(depths)
                depth_range = max_depth - min_depth
                
                # Draw horizontal grid lines
                for i in range(self.grid_resolution + 1):
                    row_points = grid_points[i*(self.grid_resolution+1):(i+1)*(self.grid_resolution+1)]
                    row_depths = [self.grid_points_3d[i*(self.grid_resolution+1)+j][2] for j in range(self.grid_resolution+1)]
                    
                    for j in range(len(row_points)-1):
                        pt1 = tuple(np.int32(row_points[j]))
                        pt2 = tuple(np.int32(row_points[j+1]))
                        
                        # Color lines based on average depth
                        avg_depth = (row_depths[j] + row_depths[j+1]) / 2
                        line_color = self.get_depth_color(avg_depth, min_depth, max_depth)
                        cv2.line(image, pt1, pt2, line_color, 1)
                
                # Draw vertical grid lines
                for j in range(self.grid_resolution + 1):
                    col_points = [grid_points[i*(self.grid_resolution+1)+j] for i in range(self.grid_resolution+1)]
                    col_depths = [self.grid_points_3d[i*(self.grid_resolution+1)+j][2] for i in range(self.grid_resolution+1)]
                    
                    for i in range(len(col_points)-1):
                        pt1 = tuple(np.int32(col_points[i]))
                        pt2 = tuple(np.int32(col_points[i+1]))
                        
                        # Color lines based on average depth
                        avg_depth = (col_depths[i] + col_depths[i+1]) / 2
                        line_color = self.get_depth_color(avg_depth, min_depth, max_depth)
                        cv2.line(image, pt1, pt2, line_color, 1)
                
                # Draw nearest grid point for hand interaction if available
                if self.use_grid_depth and self.nearest_grid_point is not None and self.debug_mode:
                    x, y, z = self.nearest_grid_point
                    cv2.circle(image, (int(x), int(y)), 8, (255, 0, 255), 2)  # Magenta circle
                    cv2.putText(image, f"Target: {int(z)}mm", (int(x)+5, int(y)+20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)
                
                # Draw depth values at grid intersections
                if self.show_depth:
                    for i, (x, y, z) in enumerate(self.grid_points_3d):
                        # Apply different display modes to control density of depth labels
                        show_this_point = False
                        
                        if self.depth_display_mode == 0:
                            # All points
                            show_this_point = True
                        elif self.depth_display_mode == 1:
                            # Alternate points (checkerboard pattern)
                            row = i // (self.grid_resolution + 1)
                            col = i % (self.grid_resolution + 1)
                            show_this_point = (row + col) % 2 == 0
                        elif self.depth_display_mode == 2:
                            # Minimal points (just corners and center)
                            row = i // (self.grid_resolution + 1)
                            col = i % (self.grid_resolution + 1)
                            is_corner = (row == 0 and col == 0) or \
                                        (row == 0 and col == self.grid_resolution) or \
                                        (row == self.grid_resolution and col == 0) or \
                                        (row == self.grid_resolution and col == self.grid_resolution)
                            is_center = row == self.grid_resolution // 2 and col == self.grid_resolution // 2
                            show_this_point = is_corner or is_center
                        
                        if show_this_point:
                            # Get color based on depth
                            depth_color = self.get_depth_color(z, min_depth, max_depth)
                            
                            # Create a darker background rectangle for text readability
                            text = f"{int(z)}"
                            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
                            cv2.rectangle(image, 
                                        (int(x)-text_size[0]//2-2, int(y)-text_size[1]//2-2), 
                                        (int(x)+text_size[0]//2+2, int(y)+text_size[1]//2+2),
                                        (0, 0, 0), -1)
                            
                            # Draw depth value centered on the point with color indicating depth
                            cv2.putText(image, text, 
                                    (int(x)-text_size[0]//2, int(y)+text_size[1]//2), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, depth_color, 1)
                
                # Show wall angle information
                if self.wall_plane is not None and hasattr(self, 'wall_angle'):
                    angle_text = f"Wall angle: {int(self.wall_angle)}°"
                    cv2.putText(image, angle_text, (20, 240), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                
                # Add depth color legend
                if self.show_depth and len(self.grid_points_3d) > 0:
                    depths = [point[2] for point in self.grid_points_3d]
                    min_depth = min(depths)
                    max_depth = max(depths)
                    
                    # Draw legend
                    legend_width = 150
                    legend_height = 20
                    legend_x = image.shape[1] - legend_width - 10
                    legend_y = image.shape[0] - legend_height - 30
                    
                    # Draw gradient bar
                    for i in range(legend_width):
                        normalized_depth = i / legend_width
                        depth_value = min_depth + normalized_depth * (max_depth - min_depth)
                        color = self.get_depth_color(depth_value, min_depth, max_depth)
                        cv2.line(image, (legend_x + i, legend_y), (legend_x + i, legend_y + legend_height), color, 1)
                    
                    # Draw min and max labels
                    cv2.putText(image, f"{int(min_depth)}mm", (legend_x - 5, legend_y + legend_height + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(image, f"{int(max_depth)}mm", (legend_x + legend_width - 40, legend_y + legend_height + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(image, "Depth", (legend_x + legend_width//2 - 20, legend_y - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show drawing mode status
                if self.drawing_mode:
                    color_name = "Custom"
                    for name, color in self.color_options:
                        if color == self.drawing_color:
                            color_name = name
                            break
                            
                    draw_info = f"Drawing: {color_name}, Thickness: {self.drawing_thickness}"
                    cv2.putText(image, draw_info, (20, image.shape[0]-20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.drawing_color, 1, cv2.LINE_AA)

        return image

    def stop_all(self):
        # Stop all operations
        self.capturing = False
        self.mapping_active = False
        self.calibration_mode = False
        
        # Close windows
        if hasattr(self, 'screen_window'):
            try:
                self.screen_window.destroy()
                del self.screen_window
            except:
                pass
                
        if hasattr(self, 'grid_window'):
            try:
                self.grid_window.destroy()
                del self.grid_window
            except:
                pass
                
        self.stop_button.config(state=tk.DISABLED)
        print("Stopped all operations.")

# Main function
if __name__ == "__main__":
    root = tk.Tk()
    app = ScreenFieldControlApp(root)
    root.mainloop()
