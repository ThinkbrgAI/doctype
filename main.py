import tkinter as tk
from src.document_classifier_gui import DocumentClassifierGUI
from src.splash_screen import SplashScreen
import sys
import os

def main():
    # Create root window but don't show it yet
    root = tk.Tk()
    root.withdraw()
    
    # Show splash screen
    splash_root = tk.Tk()
    splash = SplashScreen(splash_root)
    
    try:
        # Update splash with loading steps
        splash.update_message("Initializing application...")
        
        # Create main application
        app = DocumentClassifierGUI(root)
        
        # Close splash screen
        splash.close()
        
        # Show main window
        root.deiconify()
        root.mainloop()
        
    except Exception as e:
        import traceback
        error_msg = f"Error starting application:\n{str(e)}\n\n{traceback.format_exc()}"
        tk.messagebox.showerror("Startup Error", error_msg)
        sys.exit(1)

if __name__ == "__main__":
    main() 