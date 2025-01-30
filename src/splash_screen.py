import tkinter as tk
from tkinter import ttk
import os

class SplashScreen:
    def __init__(self, root):
        self.root = root
        self.root.overrideredirect(True)  # Remove window decorations
        
        # Calculate center position
        ws = self.root.winfo_screenwidth()
        hs = self.root.winfo_screenheight()
        x = (ws/2) - (400/2)
        y = (hs/2) - (200/2)
        
        self.root.geometry('%dx%d+%d+%d' % (400, 200, x, y))
        
        # Create frame
        frame = ttk.Frame(self.root)
        frame.pack(fill='both', expand=True)
        
        # Add logo/title
        title = ttk.Label(frame, text="DocType Classifier", font=('Helvetica', 16, 'bold'))
        title.pack(pady=20)
        
        # Add loading message
        self.message = ttk.Label(frame, text="Loading...", font=('Helvetica', 10))
        self.message.pack(pady=10)
        
        # Add progress bar
        self.progress = ttk.Progressbar(frame, length=300, mode='indeterminate')
        self.progress.pack(pady=20)
        self.progress.start()
        
        # Force update
        self.root.update()

    def update_message(self, message):
        self.message.config(text=message)
        self.root.update()

    def close(self):
        self.root.destroy() 