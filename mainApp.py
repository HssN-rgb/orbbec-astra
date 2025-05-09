import tkinter as tk
from tkinter import messagebox
from calibration_VirtualMouse import OrbbecCalibrator
import threading

class OrbbecGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Orbbec Touchscreen Calibration")

        self.calibrator = OrbbecCalibrator(grid_size=(3, 3), depth_margin=150)
        self.calibrated = False

        self.calibrate_button = tk.Button(root, text="Start Calibration", command=self.start_calibration_thread)
        self.calibrate_button.pack(pady=10)

        self.feedback_button = tk.Button(root, text="Start Feedback", command=self.start_feedback_thread, state=tk.DISABLED)
        self.feedback_button.pack(pady=10)

        self.quit_button = tk.Button(root, text="Exit", command=self.quit_app)
        self.quit_button.pack(pady=10)

    def start_calibration_thread(self):
        threading.Thread(target=self.start_calibration, daemon=True).start()

    def start_feedback_thread(self):
        threading.Thread(target=self.start_feedback, daemon=True).start()

    def start_calibration(self):
        self.calibrator.initialize_camera()
        self.calibrator.select_corners()

        if len(self.calibrator.points) == 4:
            self.calibrator.compute_grid_and_reference_depths()
            self.calibrated = True
            self.feedback_button.config(state=tk.NORMAL)
            messagebox.showinfo("Calibration", "Calibration successful.")
        else:
            messagebox.showerror("Calibration", "Calibration failed. 4 points not selected.")
            self.calibrator.shutdown_camera()

    def start_feedback(self):
        if not self.calibrated:
            messagebox.showerror("Error", "Please calibrate first.")
            return

        self.calibrator.live_feedback_loop()
        self.calibrator.shutdown_camera()

    def quit_app(self):
        try:
            self.calibrator.shutdown_camera()
        except:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = OrbbecGUI(root)
    root.mainloop()
