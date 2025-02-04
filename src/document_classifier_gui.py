import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
from pathlib import Path
from .document_classifier import DocumentClassifier
from .config import SecureConfig, setup_api_key
from openai import OpenAI
import threading
import queue
import pandas as pd
import shutil

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
        self.processing = False  # Flag to track processing state
    
    def setup_ui(self):
        """Setup the main UI components"""
        self.root.title("Document Classifier")
        self.root.geometry("800x600")
        self.root.resizable(True, True)
        
        # Configure root grid to allow expansion
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
        # Create main frame with padding
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.grid_columnconfigure(1, weight=1)
        
        # Configure main_frame rows - make log frame expandable
        for i in range(7):  # Adjust based on total rows
            weight = 1 if i == 6 else 0  # Row 6 is the log frame
            main_frame.grid_rowconfigure(i, weight=weight)
        
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
        
        # Add cost tracking frame
        cost_frame = ttk.LabelFrame(main_frame, text="Cost Tracking", padding="5")
        cost_frame.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(cost_frame, text="Total Cost:").grid(row=0, column=0, sticky=tk.W)
        self.cost_label = ttk.Label(cost_frame, text="$0.00")
        self.cost_label.grid(row=0, column=1, sticky=tk.W)
        
        ttk.Label(cost_frame, text="Tokens Used:").grid(row=1, column=0, sticky=tk.W)
        self.tokens_label = ttk.Label(cost_frame, text="0 in / 0 out")
        self.tokens_label.grid(row=1, column=1, sticky=tk.W)
        
        # Add buttons frame for Start, Stop, Export, and Organize
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=5, column=0, columnspan=3, pady=10)
        
        # Start button
        self.start_button = ttk.Button(buttons_frame, text="Start Classification", command=self.start_classification)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        # Stop button (initially disabled)
        self.stop_button = ttk.Button(buttons_frame, text="Stop Processing", command=self.stop_processing, state='disabled')
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Export button (initially disabled)
        self.export_button = ttk.Button(buttons_frame, text="Export Results", command=self.export_results, state='disabled')
        self.export_button.pack(side=tk.LEFT, padx=5)
        
        # Organize Files button (initially disabled)
        self.organize_button = ttk.Button(buttons_frame, text="Organize Files", command=self.organize_files, state='disabled')
        self.organize_button.pack(side=tk.LEFT, padx=5)
        
        # Status label
        self.status_label = ttk.Label(main_frame, text="")
        self.status_label.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E))
        
        # Log area
        log_frame = ttk.LabelFrame(main_frame, text="Log", padding="5")
        log_frame.grid(row=7, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.grid_columnconfigure(0, weight=1)
        log_frame.grid_rowconfigure(0, weight=1)
        
        # Create text widget and scrollbar
        self.log_text = tk.Text(log_frame, wrap=tk.WORD)  # Removed fixed height
        scrollbar = ttk.Scrollbar(log_frame, orient="vertical", command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=scrollbar.set)
        
        # Grid log components with sticky
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Message queue for thread communication
        self.queue = queue.Queue()
        self.root.after(100, self.check_queue)
        
        # Add API key management button
        api_frame = ttk.Frame(main_frame)
        api_frame.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(api_frame, text="Change API Key", command=self.prompt_api_key).pack(side=tk.RIGHT)
        
        # Add reorganize frame after API key management
        reorg_frame = ttk.LabelFrame(main_frame, text="Reorganize from Excel", padding="5")
        reorg_frame.grid(row=9, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Import Excel button
        ttk.Button(reorg_frame, text="Import Excel", command=self.import_excel).pack(side=tk.LEFT, padx=5)
        
        # Reorganize button (initially disabled)
        self.reorganize_button = ttk.Button(reorg_frame, text="Reorganize Files", 
            command=self.reorganize_from_excel, state='disabled')
        self.reorganize_button.pack(side=tk.LEFT, padx=5)

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
        """Start the classification process"""
        input_path = self.input_path_var.get()
        if not input_path:
            self.add_log_message("Error: Please select an input folder", error=True)
            return
            
        # Create output folder
        output_path = Path(input_path) / "output"
        output_path.mkdir(exist_ok=True)
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        
        self.processing = True
        self.start_button.state(['disabled'])
        self.stop_button.state(['!disabled'])
        self.export_button.state(['disabled'])
        self.organize_button.state(['disabled'])
        
        # Start processing in a new thread
        thread = threading.Thread(target=self.process_documents, args=(
            input_path,
            str(output_path)
        ))
        thread.daemon = True
        thread.start()

    def stop_processing(self):
        """Safely stop the classification process"""
        self.processing = False
        self.stop_button.state(['disabled'])
        self.queue.put(("status", "Stopping... Please wait for current file to complete."))
        
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
            # Convert input_path to Path object for consistent handling
            input_path = Path(input_path)
            
            # Get only PDF files with their relative paths
            files = [f for f in Path(input_path).glob("**/*") if f.is_file() and f.suffix.lower() == '.pdf']
            if not files:
                self.queue.put(("error", "No PDF files found in the selected folder"))
                self.start_button.state(['!disabled'])
                return

            total_files = len(files)
            self.queue.put(("max", ("overall", total_files)))
            
            # Clear previous results
            self.results = []
            
            total_cost = 0.0
            total_input_tokens = 0
            total_output_tokens = 0
            
            for i, file_path in enumerate(files, 1):
                if not self.processing:
                    self.queue.put(("status", "Processing stopped by user"))
                    break
                    
                # Get relative path from input directory
                rel_path = file_path.relative_to(input_path)
                self.queue.put(("status", f"Processing: {rel_path}"))
                self.queue.put(("progress", ("overall", i)))
                
                try:
                    classification = classifier.classify_document(str(file_path))
                    
                    # Update totals
                    total_cost += classification['cost']
                    total_input_tokens += classification['input_tokens']
                    total_output_tokens += classification['output_tokens']
                    
                    # Update cost display
                    self.queue.put(("cost_update", (
                        total_cost,
                        total_input_tokens,
                        total_output_tokens
                    )))
                    
                    # Store the result with full relative path
                    self.results.append({
                        'Filepath': str(rel_path),
                        'Filename': file_path.name,
                        'Directory': str(rel_path.parent),
                        'Category': classification['category'],
                        'Category Confidence': f"{classification['category_confidence']:.1f}%",
                        'Subcategory': classification['subcategory'],
                        'Subcategory Confidence': f"{classification['subcategory_confidence']:.1f}%"
                    })
                    
                    # Update GUI with result
                    self.queue.put(("file_result", (
                        str(rel_path),  # Show full relative path in log
                        classification['category'],
                        classification['subcategory'],
                        classification['category_confidence'],
                        classification['subcategory_confidence']
                    )))
                    
                except Exception as e:
                    self.queue.put(("error", f"Error processing {rel_path}: {str(e)}"))
                    self.results.append({
                        'Filepath': str(rel_path),
                        'Filename': file_path.name,
                        'Directory': str(rel_path.parent),
                        'Category': 'Error',
                        'Category Confidence': '0%',
                        'Subcategory': 'Error',
                        'Subcategory Confidence': '0%'
                    })
                
                processed = i
                self.queue.put(("progress", ("overall", processed)))
            
            # Enable export and organize buttons when processing is complete or stopped
            self.export_button.state(['!disabled'])
            self.organize_button.state(['!disabled'])
            self.start_button.state(['!disabled'])
            self.stop_button.state(['disabled'])
            
            if self.processing:
                self.queue.put(("complete", None))
            else:
                self.queue.put(("status", "Processing stopped. You can export partial results."))
            
        except Exception as e:
            self.queue.put(("error", f"Classification error: {str(e)}"))
            self.start_button.state(['!disabled'])
            self.stop_button.state(['disabled'])
            
        finally:
            self.processing = False

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
                    filepath, category, subcategory, cat_conf, subcat_conf = data
                    self.add_log_message(
                        f"Classified {filepath}:\n"
                        f"  Category: {category} ({cat_conf:.1f}%)\n"
                        f"  Subcategory: {subcategory} ({subcat_conf:.1f}%)"
                    )
                elif msg_type == "cost_update":
                    total_cost, input_tokens, output_tokens = data
                    self.cost_label.config(text=f"${total_cost:.2f}")
                    self.tokens_label.config(text=f"{input_tokens:,} in / {output_tokens:,} out")
                
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
                
            # Create DataFrame with new columns
            df = pd.DataFrame(self.results)
            
            # Add summary sheet with category counts
            with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
                # Write detailed results
                df.to_excel(writer, sheet_name='Detailed Results', index=False)
                
                # Create summary by category
                category_summary = df.groupby('Category').size().reset_index(name='Count')
                category_summary.to_excel(writer, sheet_name='Category Summary', index=False)
                
                # Create summary by subcategory
                subcategory_summary = df.groupby(['Category', 'Subcategory']).size().reset_index(name='Count')
                subcategory_summary.to_excel(writer, sheet_name='Subcategory Summary', index=False)
            
            messagebox.showinfo("Success", f"Results exported successfully to:\n{file_path}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Error exporting results:\n{str(e)}")

    def organize_files(self):
        """Copy files to categorized folder structure"""
        if not self.results:
            messagebox.showwarning("No Results", "No classification results to organize.")
            return
            
        try:
            # Get input and output paths
            input_path = Path(self.input_path_var.get())
            output_base = input_path / "organized_files"
            
            # Confirm with user
            if messagebox.askyesno("Organize Files", 
                f"This will create a structured folder hierarchy in:\n{output_base}\n\nContinue?"):
                
                # Track progress
                total_files = len(self.results)
                copied_files = 0
                errors = []
                
                for result in self.results:
                    try:
                        # Create category and subcategory folders with sanitized names
                        category_folder = output_base / self._sanitize_path(result['Category'])
                        if result['Subcategory'] != 'No Subcategory':
                            target_folder = category_folder / self._sanitize_path(result['Subcategory'])
                        else:
                            target_folder = category_folder
                            
                        # Create folder structure
                        target_folder.mkdir(parents=True, exist_ok=True)
                        
                        # Source and target file paths
                        source_file = input_path / result['Filepath']
                        
                        # Sanitize filename while preserving extension
                        filename = Path(result['Filename'])
                        safe_filename = self._sanitize_path(filename.stem) + filename.suffix
                        target_file = target_folder / safe_filename
                        
                        # Handle duplicate filenames
                        counter = 1
                        base_name = target_file.stem
                        extension = target_file.suffix
                        while target_file.exists():
                            target_file = target_folder / f"{base_name}_{counter}{extension}"
                            counter += 1
                        
                        # Copy file using absolute paths
                        import shutil
                        shutil.copy2(str(source_file.absolute()), str(target_file.absolute()))
                        copied_files += 1
                        
                        # Update status
                        self.status_label.config(text=f"Copying files: {copied_files}/{total_files}")
                        
                    except Exception as e:
                        errors.append(f"Error copying {result['Filepath']}: {str(e)}")
                        self.add_log_message(f"Error copying {result['Filepath']}: {str(e)}", error=True)
                
                # Show completion message
                if errors:
                    messagebox.showwarning("Organization Complete with Errors",
                        f"Copied {copied_files} of {total_files} files.\n\nErrors:\n" + "\n".join(errors))
                else:
                    messagebox.showinfo("Organization Complete",
                        f"Successfully copied {copied_files} files to categorized folders.")
                
                # Reset status
                self.status_label.config(text="File organization complete")
                
        except Exception as e:
            messagebox.showerror("Organization Error", f"Error organizing files:\n{str(e)}")

    def _sanitize_path(self, path_str):
        """Convert string to valid path name"""
        # Map long category names to shorter versions
        category_map = {
            'Pay Applications and Job Cost Information': 'Pay Apps',
            'Contemporaneous Documentation': 'Contemp Docs',
            'Inspection Reports and Punchlists': 'Inspections',
            'Daily Reports / Field Reports': 'Daily Reports',
            'Plans & Specifications': 'Plans-Specs',
            'Key Dates and Schedules': 'Schedules',
            'Contracts and Changes': 'Contracts'
        }
        
        # Remove any GPT commentary (text after quotes or periods)
        if '"' in path_str:
            path_str = path_str.split('"')[0]
        if '. ' in path_str:
            path_str = path_str.split('. ')[0]
        
        # Clean up the string
        path_str = path_str.strip()
        
        # Try to map to shorter name first
        sanitized = category_map.get(path_str, path_str)
        
        # Remove or replace invalid characters
        invalid_chars = '<>:"/\\|?*[](){}!@#$%^&+=`~.'
        
        # Replace invalid characters with underscores
        for char in invalid_chars:
            sanitized = sanitized.replace(char, '_')
        
        # Replace multiple spaces/underscores with single underscore and remove trailing spaces
        sanitized = '_'.join(word.strip() for word in sanitized.split())
        while '__' in sanitized:
            sanitized = sanitized.replace('__', '_')
        
        # Remove leading/trailing underscores and spaces
        sanitized = sanitized.strip('_ ')
        
        # Enforce maximum length (without trailing spaces)
        if len(sanitized) > 30:
            sanitized = sanitized[:27].rstrip() + '...'
            
        # Final cleanup to ensure no trailing spaces
        sanitized = sanitized.rstrip()
            
        return sanitized

    def import_excel(self):
        """Import classification results from modified Excel file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Excel Results File",
                filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            
            if not file_path:
                return
                
            # Read Excel file
            df = pd.read_excel(file_path, sheet_name='Detailed Results')
            
            # Validate required columns
            required_cols = ['Filepath', 'Category', 'Subcategory']
            if not all(col in df.columns for col in required_cols):
                raise ValueError("Excel file must contain columns: Filepath, Category, Subcategory")
            
            # Convert to results format
            self.results = df.to_dict('records')
            
            # Enable reorganize button
            self.reorganize_button.state(['!disabled'])
            
            # Show summary of changes
            categories = df.groupby('Category').size()
            self.add_log_message("\nImported classifications:")
            for cat, count in categories.items():
                self.add_log_message(f"{cat}: {count} files")
            
            messagebox.showinfo("Import Success", 
                f"Successfully imported {len(df)} classifications.\nYou can now reorganize files based on these changes.")
            
        except Exception as e:
            messagebox.showerror("Import Error", f"Error importing Excel file:\n{str(e)}")
            self.reorganize_button.state(['disabled'])

    def reorganize_from_excel(self):
        """Reorganize files based on imported Excel classifications"""
        if not self.results:
            messagebox.showwarning("No Data", "Please import an Excel file first.")
            return
            
        try:
            input_path = Path(filedialog.askdirectory(
                title="Select Source Folder (containing original files)"
            ))
            
            if not input_path:
                return
                
            # Use shorter base folder name with new name
            output_base = input_path / "re-organized_files"  # Changed from "classified_docs"
            output_base.mkdir(parents=True, exist_ok=True)
            
            if messagebox.askyesno("Reorganize Files", 
                f"This will reorganize files into:\n{output_base}\n\nContinue?"):
                
                total_files = len(self.results)
                copied_files = 0
                errors = []
                
                # Create all target folders first
                unique_categories = {(r['Category'], r['Subcategory']) for r in self.results}
                for category, subcategory in unique_categories:
                    try:
                        category_folder = output_base / self._sanitize_path(category)[:20]
                        category_folder.mkdir(exist_ok=True)
                        
                        if subcategory != 'No Subcategory':
                            subcat_folder = category_folder / self._sanitize_path(subcategory)[:20]
                            subcat_folder.mkdir(exist_ok=True)
                    except Exception as e:
                        self.add_log_message(f"Error creating folder for {category}/{subcategory}: {e}", error=True)
                
                # Add debug list for problematic files
                problem_files = ['TCI_00000347.pdf', 'TCI_00000256.pdf', 'TCI_00000315.pdf']
                
                for result in self.results:
                    try:
                        # Setup target folders with debug info for problem files
                        is_problem_file = Path(result['Filepath']).name in problem_files
                        
                        if is_problem_file:
                            self.add_log_message("\nDEBUG - Target Path Creation:")
                            self.add_log_message(f"Category: {result['Category']}")
                            self.add_log_message(f"Sanitized Category: {self._sanitize_path(result['Category'])[:20]}")
                            self.add_log_message(f"Subcategory: {result['Subcategory']}")
                            self.add_log_message(f"Sanitized Subcategory: {self._sanitize_path(result['Subcategory'])[:20]}")
                        
                        # Create target folder path
                        category_folder = output_base / self._sanitize_path(result['Category'])[:20]
                        if result['Subcategory'] != 'No Subcategory':
                            target_folder = category_folder / self._sanitize_path(result['Subcategory'])[:20]
                        else:
                            target_folder = category_folder
                        
                        if is_problem_file:
                            self.add_log_message(f"Target folder path: {target_folder}")
                            self.add_log_message(f"Target folder exists: {target_folder.exists()}")
                            self.add_log_message(f"Parent exists: {target_folder.parent.exists()}")
                            try:
                                self.add_log_message(f"Can write to folder: {os.access(str(target_folder.parent), os.W_OK)}")
                            except Exception as e:
                                self.add_log_message(f"Error checking write access: {e}")
                        
                        # Ensure target folder exists
                        try:
                            target_folder.mkdir(parents=True, exist_ok=True)
                            if is_problem_file:
                                self.add_log_message("Successfully created target folder")
                        except Exception as e:
                            if is_problem_file:
                                self.add_log_message(f"Error creating target folder: {e}")
                            raise
                        
                        # Find source file using multiple strategies
                        filepath = result['Filepath'].replace('\\', '/')
                        filename = Path(filepath).name
                        
                        # Extra debugging for problem files
                        is_problem_file = filename in problem_files
                        if is_problem_file:
                            self.add_log_message(f"\nDEBUG for {filename}:")
                            self.add_log_message(f"Input path: {input_path}")
                            self.add_log_message(f"Looking for file: {filepath}")
                            self.add_log_message(f"Exists in Excel: {filename in [Path(r['Filepath']).name for r in self.results]}")
                        
                        # Search strategies with detailed logging
                        source_file = None
                        search_methods = [
                            (lambda: input_path / filepath, "Full relative path"),
                            (lambda: input_path / filename, "Direct filename"),
                            (lambda: next(iter(input_path.glob(f"**/{filename}")), None), "Recursive glob"),
                            (lambda: next((p for p in input_path.rglob(filename) 
                                        if str(p).lower().endswith(filename.lower())), None), "Case-insensitive")
                        ]
                        
                        for find_method, method_name in search_methods:
                            try:
                                found_path = find_method()
                                if is_problem_file:
                                    self.add_log_message(f"Trying {method_name}: {found_path}")
                                    if found_path:
                                        self.add_log_message(f"Path exists: {found_path.exists()}")
                                        self.add_log_message(f"Is file: {found_path.is_file() if found_path.exists() else 'N/A'}")
                                
                                if found_path and found_path.is_file():
                                    source_file = found_path
                                    if is_problem_file:
                                        self.add_log_message(f"Found file using {method_name}")
                                    break
                            except Exception as e:
                                if is_problem_file:
                                    self.add_log_message(f"Error with {method_name}: {str(e)}")
                                continue
                        
                        # Try one last direct attempt for problem files
                        if not source_file and is_problem_file:
                            try:
                                # Try exact path
                                exact_path = input_path / "Timmons doc production 12.13.24" / "TCI 1-399" / filename
                                self.add_log_message(f"Trying exact path: {exact_path}")
                                self.add_log_message(f"Exists: {exact_path.exists()}")
                                if exact_path.is_file():
                                    source_file = exact_path
                                    self.add_log_message("Found using exact path")
                                
                                # List contents of parent directory
                                parent_dir = exact_path.parent
                                if parent_dir.exists():
                                    self.add_log_message(f"\nContents of {parent_dir}:")
                                    for f in parent_dir.iterdir():
                                        self.add_log_message(f"  {f.name}")
                            except Exception as e:
                                self.add_log_message(f"Error checking exact path: {str(e)}")
                        
                        if not source_file:
                            raise FileNotFoundError(f"Could not find source file: {filepath}")
                        
                        # Create target file path with debug info
                        target_file = target_folder / filename
                        
                        if is_problem_file:
                            self.add_log_message("\nDEBUG - File Copy:")
                            self.add_log_message(f"Source file: {source_file}")
                            self.add_log_message(f"Target file: {target_file}")
                            self.add_log_message(f"Source exists: {source_file.exists()}")
                            self.add_log_message(f"Source is file: {source_file.is_file()}")
                            self.add_log_message(f"Target folder exists: {target_folder.exists()}")
                        
                        # Handle duplicates
                        counter = 1
                        while target_file.exists():
                            target_file = target_folder / f"{target_file.stem}_{counter}{target_file.suffix}"
                            counter += 1
                            if is_problem_file:
                                self.add_log_message(f"Duplicate handling - new target: {target_file}")
                        
                        # Copy file with explicit error handling
                        try:
                            if is_problem_file:
                                self.add_log_message("Attempting to copy file...")
                            shutil.copy2(str(source_file), str(target_file))
                            if is_problem_file:
                                self.add_log_message("File copied successfully")
                        except Exception as e:
                            if is_problem_file:
                                self.add_log_message(f"Copy operation failed: {str(e)}")
                                self.add_log_message(f"Source path length: {len(str(source_file))}")
                                self.add_log_message(f"Target path length: {len(str(target_file))}")
                            raise
                        copied_files += 1
                        self.status_label.config(text=f"Reorganizing: {copied_files}/{total_files}")
                        
                    except Exception as e:
                        errors.append(f"Error copying {result['Filepath']}: {str(e)}")
                        self.add_log_message(f"Error copying {result['Filepath']}: {str(e)}", error=True)
                
                # Show completion message
                if errors:
                    messagebox.showwarning("Reorganization Complete with Errors",
                        f"Copied {copied_files} of {total_files} files.\n\nErrors:\n" + "\n".join(errors))
                else:
                    messagebox.showinfo("Reorganization Complete",
                        f"Successfully reorganized {copied_files} files according to Excel classifications.")
                
                self.status_label.config(text="File reorganization complete")
                
        except Exception as e:
            messagebox.showerror("Reorganization Error", f"Error reorganizing files:\n{str(e)}")

def main():
    root = tk.Tk()
    app = DocumentClassifierGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main() 