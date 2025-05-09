import time
import autopy
import cv2
import numpy as np
from openni import openni2
import bodyTrackingModule as btm
import monitorSelectionModule as msm

class OrbbecCalibrator:
    def __init__(self, grid_size=(4, 4), depth_margin=150, median_filter_size=5):
        self.grid_size = grid_size
        self.depth_margin = depth_margin
        self.median_filter_size = median_filter_size  # Size of the median filter
        self.points = []
        self.virtual_points = []
        self.reference_depths = []
        self.homography = None
        self.running = False
        self.device = None
        self.color_stream = None
        self.depth_stream = None

    def initialize_camera(self):
        openni2.initialize()
        self.device = openni2.Device.open_any()

        self.color_stream = self.device.create_color_stream()
        self.depth_stream = self.device.create_depth_stream()

        self.color_stream.start()
        self.depth_stream.start()

    def shutdown_camera(self):
        self.color_stream.stop()
        self.depth_stream.stop()
        openni2.unload()
        cv2.destroyAllWindows()

    def get_color_frame(self):
        frame = self.color_stream.read_frame()
        data = np.frombuffer(frame.get_buffer_as_uint8(), dtype=np.uint8).reshape((frame.height, frame.width, 3))
        return cv2.cvtColor(data.copy(), cv2.COLOR_RGB2BGR)

    def get_depth_frame(self):
        frame = self.depth_stream.read_frame()
        data = np.frombuffer(frame.get_buffer_as_uint16(), dtype=np.uint16).reshape((frame.height, frame.width))
        return data.copy()

    def clamp_depth(self, depth_value):
        """Clamp invalid depth values outside of the valid range (500mm to 4000mm)."""
        if depth_value < 500:
            return 500
        elif depth_value > 4000:
            return 4000
        return depth_value

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.points) < 4:
            self.points.append((x, y))
            print(f"[INFO] Selected point {len(self.points)}: ({x}, {y})")

    def select_corners(self):
        print("[INFO] Please select 4 corners (clockwise or counter-clockwise)...")
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", self.mouse_callback)

        while len(self.points) < 4:
            frame = self.get_color_frame()
            for pt in self.points:
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)
            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyWindow("Calibration")

    def compute_grid_and_reference_depths(self):
        width, height = self.grid_size
        src_pts = np.array(self.points, dtype=np.float32)
        dst_pts = np.array([
            [0, 0], [width - 1, 0],
            [width - 1, height - 1], [0, height - 1]
        ], dtype=np.float32)

        self.homography = cv2.getPerspectiveTransform(src_pts, dst_pts)
        self.virtual_points = []
        self.reference_depths = []

        # Warm up depth sensor
        for _ in range(10):
            self.get_depth_frame()

        depth_frame = self.get_depth_frame()

        # Generate a 4x4 grid in normalized (virtual) space
        for j in range(height):
            for i in range(width):
                norm_pt = np.array([[i, j]], dtype=np.float32)
                screen_pt = cv2.perspectiveTransform(np.array([norm_pt]), np.linalg.inv(self.homography))[0][0]
                sx, sy = int(screen_pt[0]), int(screen_pt[1])

                # Clamp to valid coordinates
                if 0 <= sx < depth_frame.shape[1] and 0 <= sy < depth_frame.shape[0]:
                    depth = self.clamp_depth(depth_frame[sy, sx])
                else:
                    depth = 1000  # Default if out of bounds

                self.virtual_points.append((sx, sy))
                self.reference_depths.append(depth)

        print(f"[INFO] {width * height} grid points and reference depths computed.")

    def apply_median_filter(self, depth_frame):
        """Apply a median filter to the depth frame for noise reduction."""
        return cv2.medianBlur(depth_frame, self.median_filter_size)

    def draw_grid_and_distances(self, img, depth_frame):
        for idx, (x, y) in enumerate(self.virtual_points):
            ref_d = self.reference_depths[idx]
            d = depth_frame[y, x] if 0 <= x < depth_frame.shape[1] and 0 <= y < depth_frame.shape[0] else 0

            # Clamp or filter invalid data
            d = self.clamp_depth(d)

            distance = abs(int(d) - int(ref_d))  # Calculate the absolute distance in mm
            color = (0, 255, 0) if distance < self.depth_margin else (0, 0, 255)  # Color if within margin

            # Display the mm distance difference between the landmarks and the grid point
            cv2.putText(img, f"{distance:.2f} mm", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Draw the grid point
            cv2.circle(img, (x, y), 5, color, -1)

            # Draw small square grid cells for better visual understanding
            cv2.rectangle(img, (x - 10, y - 10), (x + 10, y + 10), color, 2)

    def live_feedback_loop(self):
        print("[INFO] Starting visual feedback with body tracking. ESC to quit.")
        detector = btm.BodyTracker()
        screen = msm.get_monitor_by_index(0)
        wScr, hScr = screen.width, screen.height

        frameR = 100
        pTime = 0

        click_cooldown = 0.5  # seconds
        last_click_time = 0

        while True:
            color_frame = self.get_color_frame()
            depth_frame = self.get_depth_frame()
            img = detector.findBody(color_frame)
            lmList = detector.findPosition(img)

            lm_dict = {id: (x, y) for id, x, y in lmList}

            for pair in [(19, 20), (20, 19)]:
                id_back, id_front = pair
                if id_back in lm_dict and id_front in lm_dict:
                    x1, y1 = lm_dict[id_front]  # For interaction
                    x2, y2 = lm_dict[id_back]  # For depth comparison

                    if not (0 <= x1 < depth_frame.shape[1] and 0 <= y1 < depth_frame.shape[0] and
                            0 <= x2 < depth_frame.shape[1] and 0 <= y2 < depth_frame.shape[0]):
                        continue

                    try:
                        depth_20 = int(depth_frame[y1, x1])
                        depth_19 = int(depth_frame[y2, x2])
                    except IndexError:
                        continue

                    if depth_20 == 0 or depth_19 == 0:
                        continue

                    closest_idx = np.argmin([
                        abs(depth_20 - ref_d) if ref_d > 0 else float('inf')
                        for ref_d in self.reference_depths
                    ])
                    grid_x, grid_y = self.virtual_points[closest_idx]
                    grid_depth = self.reference_depths[closest_idx]

                    diff_20 = abs(depth_20 - grid_depth)
                    diff_19 = abs(depth_19 - grid_depth)

                    # Draw mm differences
                    cv2.putText(img, f"20 Δ: {diff_20:.1f} mm", (x1 + 10, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    cv2.putText(img, f"19 Δ: {diff_19:.1f} mm", (x2 + 10, y2 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                    # Perform click if hand is close enough
                    if diff_20 < self.depth_margin:
                        current_time = time.time()
                        if current_time - last_click_time > click_cooldown:
                            autopy.mouse.click()
                            last_click_time = current_time
                            print("[INFO] Click triggered")
                        cv2.circle(img, (x1, y1), 10, (255, 0, 0), cv2.FILLED)
                    break  # Use only one valid hand per frame

            self.draw_grid_and_distances(img, depth_frame)

            # FPS counter
            cTime = time.time()
            fps = 1 / (cTime - pTime) if cTime != pTime else 0
            pTime = cTime
            cv2.putText(img, f'FPS: {int(fps)}', (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.imshow("Touchscreen Feedback", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break

        cv2.destroyWindow("Touchscreen Feedback")

    def start_calibration(self):
        self.initialize_camera()
        self.select_corners()
        if len(self.points) != 4:
            print("[ERROR] Calibration aborted: less than 4 points selected.")
            self.shutdown_camera()
            return

        self.compute_grid_and_reference_depths()
        self.live_feedback_loop()
        self.shutdown_camera()