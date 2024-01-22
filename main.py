import tkinter as tk
import os
from screenshot_selector import ScreenshotSelector
import screenshot_selector


class MainApplication:
    def __init__(self, root, data_path='./data', start_idx=0):
        self.root = root
        self.idx = start_idx
        self.data_path = data_path

        self.root.geometry('350x250')  # Size of the main window

        # Button to take a screenshot
        self.start_button = tk.Button(root, text='截棋盘图',
                                      command=lambda: self.on_click_start(mode='board'))
        self.start_button.pack(pady=20)

        # Button to take a screenshot
        self.start_button1 = tk.Button(root, text='截解释图',
                                       command=lambda: self.on_click_start(mode='expl'))
        self.start_button1.pack(pady=20)

        self.start_button2 = tk.Button(root, text='下一个',
                                       command=lambda: self.on_click_next())
        self.start_button2.pack(pady=20)

        self.index_str = tk.StringVar()
        self.index_str.set(f"Current data index: {self.idx}")
        self.idx_label = tk.Label(root, textvariable=self.index_str)
        self.idx_label.pack()

        # Area to display the screenshot
        self.display_area = tk.Label(root, text='棋盘图')
        self.display_area.pack(expand=True)
        self.display_area1 = tk.Label(root, text='解释图')
        self.display_area1.pack(expand=True)

        self.start_button.place(x=50, y=30)
        self.start_button1.place(x=140, y=30)
        self.start_button2.place(x=230, y=30)
        self.display_area.place(x=20, y=80)
        self.display_area1.place(x=200, y=80)

    def on_click_start(self, mode):
        dp = os.path.join(self.data_path, f'{self.idx}_{mode}.png')
        ScreenshotSelector(self, dp, mode)

    def on_click_next(self):
        self.idx += 1
        self.index_str.set(f"Current data index: {self.idx}")
        self.display_area.config(image=None)
        self.display_area.image = None  # Keep a reference!
        self.display_area1.config(image=None)
        self.display_area1.image = None  # Keep a reference!


def main():
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()


if __name__ == "__main__":
    screenshot_selector.SCREEN_DPI = '1500x800'
    screenshot_selector.SCREEN_RATIO = 1.25
    main()
