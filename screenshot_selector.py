import tkinter as tk
from PIL import ImageGrab
from PIL import Image, ImageTk
import os


SCREEN_DPI = '1500x800'
SCREEN_RATIO = 1.25


class ScreenshotSelector:
    def __init__(self, parent, img_path, mode):
        self.parent = parent.root
        self.app = parent
        self.img_path = img_path
        self.top = tk.Toplevel(self.parent)
        self.mode = mode

        # Set the window as topmost and semi-transparent
        self.top.attributes('-topmost', True)
        self.top.attributes('-alpha', 0.3)  # Adjust transparency level

        self.top.overrideredirect(True)
        self.top.geometry(SCREEN_DPI)  # Default size for selection window

        self.canvas = tk.Canvas(self.top, cursor="cross", bg='grey')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

    def on_press(self, event):
        # Translate widget coordinates to screen coordinates
        self.start_x = self.top.winfo_rootx() + event.x
        self.start_y = self.top.winfo_rooty() + event.y
        self.selection = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline='red')

    def on_drag(self, event):
        curX, curY = (event.x, event.y)
        self.canvas.coords(self.selection, self.start_x, self.start_y, curX, curY)

    def on_release(self, event):
        # Translate widget coordinates to screen coordinates
        end_x = self.top.winfo_rootx() + event.x
        end_y = self.top.winfo_rooty() + event.y
        self.capture_screenshot(self.start_x, self.start_y, end_x, end_y)
        self.top.destroy()

    def capture_screenshot(self, start_x, start_y, end_x, end_y):
        # Account for screen scaling if necessary
        scale_factor = SCREEN_RATIO  # Adjust this based on your screen scaling settings
        x1 = int(start_x * scale_factor)
        y1 = int(start_y * scale_factor)
        x2 = int(end_x * scale_factor)
        y2 = int(end_y * scale_factor)

        x1 = min(x1, x2)
        x2 = max(x1, x2)
        y1 = min(y1, y2)
        y2 = max(y1, y2)

        screenshot = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        screenshot.save(self.img_path)
        if self.mode == 'board':
            update_screenshot_display(self.app.display_area, self.img_path)
        else:
            update_screenshot_display(self.app.display_area1, self.img_path)


def start_screenshot_selector(app, img_path, mode):
    ScreenshotSelector(app, img_path, mode)


def update_screenshot_display(display_area, filepath):
    if os.path.exists(filepath):
        img = Image.open(filepath)
        img = img.resize((120, 120), Image.ANTIALIAS)  # Resize if necessary
        photo = ImageTk.PhotoImage(img)

        # Update the display area with the new image
        display_area.config(image=photo)
        display_area.image = photo  # Keep a reference!

