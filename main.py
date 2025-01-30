from src.document_classifier_gui import DocumentClassifierGUI
import tkinter as tk

def main():
    root = tk.Tk()
    app = DocumentClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 