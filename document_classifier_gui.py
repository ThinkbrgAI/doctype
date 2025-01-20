import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
from document_classifier import DocumentClassifier
from config import SecureConfig, setup_api_key
from openai import OpenAI
import threading
import queue
import pandas as pd

class DocumentClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.withdraw()  # Hide main window initially
        
        # Check for API key first
        if not self.check_api_key():
            self.root.destroy()
            return
            
        # Now show main window and continue with UI setup
        self.root.deiconify()
        self.setup_ui()
        
        self.results = []  # Store classification results
    
    def setup_ui(self):
        """Setup the main UI components"""
        self.root.title("Document Classifier")
        self.root.geometry("800x600")  # Made window larger to accommodate log
        self.root.resizable(True, True)
        
        # Configure root grid to allow expansion
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.grid_columnconfigure(1, weight=1)
        main_frame.grid_rowconfigure(5, weight=1)  # Make log area expandable
        
        # Model selection
        ttk.Label(main_frame, text="Select Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.model_var = tk.StringVar(value="o1-2024-12-17")
        model_combo = ttk.Combobox(main_frame, textvariable=self.model_var)
        model_combo['values'] = ('o1-2024-12-17', 'chatgpt-4o-latest')
        model_combo.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Input folder selection
        ttk.Label(main_frame, text="Input Folder:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.input_path_var = tk.StringVar()
        ttk.Entry(main_frame, textvariable=self.input_path_var).grid(row=1, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_input).grid(row=1, column=2, sticky=tk.W, pady=5, padx=5)
        
        # Add prompt customization section
        prompt_frame = ttk.LabelFrame(main_frame, text="System Prompt", padding="5")
        prompt_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        prompt_frame.grid_columnconfigure(0, weight=1)
        
        self.prompt_text = tk.Text(prompt_frame, height=8, wrap=tk.WORD)
        prompt_scrollbar = ttk.Scrollbar(prompt_frame, orient="vertical", command=self.prompt_text.yview)
        self.prompt_text.configure(yscrollcommand=prompt_scrollbar.set)
        
        self.prompt_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        prompt_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Load default prompt
        self.prompt_text.insert('1.0', DocumentClassifier.DEFAULT_PROMPT)
        
        # Add reset prompt button
        button_frame = ttk.Frame(prompt_frame)
        button_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.E), pady=5)
        ttk.Button(button_frame, text="Reset to Default", command=self.reset_prompt).pack(side=tk.RIGHT)
        
        # Progress section
        progress_frame = ttk.LabelFrame(main_frame, text="Progress", padding="5")
        progress_frame.grid(row=3, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        progress_frame.grid_columnconfigure(1, weight=1)
        
        # File progress
        ttk.Label(progress_frame, text="Current File:").grid(row=0, column=0, sticky=tk.W)
        self.current_file_label = ttk.Label(progress_frame, text="")
        self.current_file_label.grid(row=0, column=1, sticky=tk.W)
        
        self.file_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.file_progress.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Overall progress
        ttk.Label(progress_frame, text="Overall Progress:").grid(row=2, column=0, sticky=tk.W)
        self.overall_progress = ttk.Progressbar(progress_frame, length=400, mode='determinate')
        self.overall_progress.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Add buttons frame for Start and Export
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=4, column=0, columnspan=3, pady=10)
        
        # Start button
        self.start_button = ttk.Button(buttons_frame, text="Start Classification", command=self.start_classification)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Export button (initially disabled)
        self.export_button = ttk.Button(buttons_frame, text="Export Results", command=self.export_results, state='disabled')
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.grid(row=5, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # Log area
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="5")
        log_frame.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)
        
        # Create text widget and scrollbar
        self.log_text = tk.Text(log_frame, height=8, wrap=tk.WORD)
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        # Grid log components
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Message queue for thread communication
        self.queue = queue.Queue()
        self.root.after(100, self.check_queue)
        
        # Add API key management button
        api_frame = ttk.Frame(main_frame)
        api_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(api_frame, text="Change API Key", command=self.prompt_api_key).pack(side=tk.RIGHT)

    def add_log_message(self, message, error=False):
        """Add message to log area with optional error formatting"""
        self.log_text.insert(tk.END, f"{message}\n")
        if error:
            # Get the last line's position
            last_line = self.log_text.get("end-2c linestart", "end-1c")
            start_pos = f"end-{len(last_line)+2}c"
            end_pos = "end-1c"
            self.log_text.tag_add("error", start_pos, end_pos)
            self.log_text.tag_config("error", foreground="red")
        self.log_text.see(tk.END)  # Auto-scroll to bottom

    def browse_input(self):
        folder_path = filedialog.askdirectory(
            title="Select Folder with PDF Documents"
        )
        if folder_path:
            self.input_path_var.set(folder_path)

    def start_classification(self):
        input_path = self.input_path_var.get()
        if not input_path:
            self.add_log_message("Error: Please select an input folder", error=True)
            return
            
        # Create output folder
        output_path = Path(input_path) / "output"
        output_path.mkdir(exist_ok=True)
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        
        # Disable start button
        self.start_button.state(['disabled'])
        
        # Start processing thread
        thread = threading.Thread(target=self.process_documents, args=(input_path, str(output_path)))
        thread.daemon = True
        thread.start()

    def process_documents(self, input_path, output_path):
        try:
            # Get API key
            config = SecureConfig()
            api_key = config.get_api_key()
            
            if not api_key:
                self.queue.put(("error", "API key not found. Please configure your API key."))
                self.start_button.state(['!disabled'])
                return
                
            if not self.validate_api_key(api_key):
                self.queue.put(("error", "Invalid API key. Please update your API key."))
                self.start_button.state(['!disabled'])
                if self.prompt_api_key():
                    # Retry with new key
                    self.start_classification()
                return
            
            # Get current prompt
            custom_prompt = self.prompt_text.get('1.0', tk.END).strip()
            if custom_prompt == DocumentClassifier.DEFAULT_PROMPT.strip():
                custom_prompt = None  # Use default if unchanged
            
            classifier = DocumentClassifier(
                api_key, 
                output_path, 
                self.model_var.get(),
                custom_prompt=custom_prompt
            )
            # Get only PDF files
            files = [f for f in Path(input_path).glob("**/*") if f.is_file() and f.suffix.lower() == '.pdf']
            if not files:
                self.queue.put(("error", "No PDF files found in the selected folder"))
                self.start_button.state(['!disabled'])
                return

            total_files = len(files)
            self.queue.put(("max", ("overall", total_files)))
            
            # Clear previous results
            self.results = []
            self.export_button.state(['disabled'])
            
            for i, file_path in enumerate(files, 1):
                self.queue.put(("status", f"Processing: {file_path.name}"))
                self.queue.put(("progress", ("overall", i)))
                
                try:
                    doc_type, confidence = classifier.classify_document(str(file_path))
                    
                    # Store the result
                    self.results.append({
                        'Filename': file_path.name,
                        'Document Type': doc_type,
                        'Confidence Score': f"{confidence:.1f}%"
                    })
                    
                    # Update GUI with result
                    self.queue.put(("file_result", (file_path.name, doc_type, confidence)))
                    
                except Exception as e:
                    self.queue.put(("error", f"Error processing {file_path.name}: {str(e)}"))
                    self.results.append({
                        'Filename': file_path.name,
                        'Document Type': 'Error',
                        'Confidence Score': '0%'
                    })
                
                processed = i
                self.queue.put(("progress", ("overall", processed)))
            
            # Enable export button when processing is complete
            self.export_button.state(['!disabled'])
            
            self.queue.put(("complete", None))
            
        except Exception as e:
            self.queue.put(("error", f"Classification error: {str(e)}"))
            self.start_button.state(['!disabled'])

    def check_queue(self):
        try:
            while True:
                msg_type, data = self.queue.get_nowait()
                
                if msg_type == "status":
                    self.status_label.config(text=data)
                    self.current_file_label.config(text=data)
                elif msg_type == "progress":
                    bar_name, value = data
                    if bar_name == "overall":
                        self.overall_progress["value"] = value
                elif msg_type == "max":
                    bar_name, value = data
                    if bar_name == "overall":
                        self.overall_progress["maximum"] = value
                elif msg_type == "complete":
                    self.status_label.config(text="Classification complete!")
                    self.start_button.state(['!disabled'])
                    self.add_log_message("Classification complete!")
                elif msg_type == "error":
                    self.add_log_message(data, error=True)
                elif msg_type == "file_result":
                    filename, doc_type, confidence = data
                    self.add_log_message(f"Classified {filename}: {doc_type} ({confidence:.1f}%)")
                
                self.queue.task_done()
                
        except queue.Empty:
            pass
        
        self.root.after(100, self.check_queue)

    def check_api_key(self):
        """Check for API key and prompt if missing"""
        config = SecureConfig()
        api_key = config.get_api_key()
        
        if not api_key:
            # Try environment variable
            api_key = os.getenv("OPENAI_API_KEY")
            
        if not api_key:
            # Prompt user for API key
            return self.prompt_api_key()
        return True

    def validate_api_key(self, key):
        """Validate the API key with OpenAI"""
        try:
            client = OpenAI(api_key=key)
            # Make a simple API call to test the key
            client.models.list()
            return True
        except Exception as e:
            return False

    def prompt_api_key(self):
        """Show dialog to get API key from user"""
        dialog = tk.Toplevel(self.root)
        dialog.title("OpenAI API Key Required")
        dialog.geometry("400x180")
        dialog.transient(self.root)
        dialog.grab_set()  # Make dialog modal
        
        # Center dialog on screen
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - dialog.winfo_width()) // 2
        y = (dialog.winfo_screenheight() - dialog.winfo_height()) // 2
        dialog.geometry(f"+{x}+{y}")
        
        # Add widgets
        ttk.Label(dialog, text="Please enter your OpenAI API key:", padding="10").pack()
        
        api_key_var = tk.StringVar()
        entry = ttk.Entry(dialog, textvariable=api_key_var, width=50, show="*")
        entry.pack(padx=10, pady=5)
        
        status_label = ttk.Label(dialog, text="", foreground="red")
        status_label.pack(pady=5)
        
        def save_key():
            key = api_key_var.get().strip()
            if not key:
                status_label.config(text="API key cannot be empty")
                return
                
            if not self.validate_api_key(key):
                status_label.config(text="Invalid API key. Please check and try again.")
                return
                
            # Use SecureConfig directly instead of setup_api_key
            config = SecureConfig()
            config.save_api_key(key)
            dialog.destroy()
        
        ttk.Button(dialog, text="Save", command=save_key).pack(pady=10)
        
        # Focus entry
        entry.focus_set()
        
        # Wait for dialog to close
        self.root.wait_window(dialog)
        
        # Check if key was saved and is valid
        config = SecureConfig()
        api_key = config.get_api_key()
        if not api_key or not self.validate_api_key(api_key):
            if self.root.winfo_exists():  # Check if root window still exists
                messagebox.showerror("Error", "Valid API key is required to use this application")
            return False
        return True

    def reset_prompt(self):
        """Reset the prompt to default"""
        self.prompt_text.delete('1.0', tk.END)
        self.prompt_text.insert('1.0', DocumentClassifier.DEFAULT_PROMPT)

    def export_results(self):
        """Export classification results to Excel"""
        if not self.results:
            messagebox.showwarning("No Results", "No classification results to export.")
            return
            
        try:
            # Ask user for save location
            file_path = filedialog.asksaveasfilename(
                defaultextension=".xlsx",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")],
                title="Save Classification Results"
            )
            
            if not file_path:  # User cancelled
                return
                
            # Create DataFrame and save to Excel
            df = pd.DataFrame(self.results)
            df.to_excel(file_path, index=False)
            
            messagebox.showinfo("Success", f"Results exported successfully to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting results:\n{str(e)}")

def main():
    root = tk.Tk()
    app = DocumentClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 