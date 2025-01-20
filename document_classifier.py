import os
import magic
import pandas as pd
from openai import OpenAI
from pathlib import Path
from pdf2image import convert_from_path
import base64
import io
import logging
from datetime import datetime
import cv2
import numpy as np
import pytesseract
from PIL import Image

class DocumentClassifier:
    # Available models with their corresponding API endpoints
    MODELS = {
        'o1-2024-12-17': 'o1-2024-12-17',  # Use exact model name
        'chatgpt-4o-latest': 'chatgpt-4o-latest'  # Use exact model name
    }
    
    # Default system prompt
    DEFAULT_PROMPT = """You are a document classification expert. Analyze the provided document and classify it into one of these categories:
1. Plans & Specifications
2. Key Dates and Schedules
3. Contracts and Changes
4. Meeting Minutes
5. Pay Applications
6. Daily Reports
7. Inspection Reports
8. Documentation
9. Miscellaneous

Return only the category number and name, followed by a confidence percentage in parentheses.
Example: "3. Contracts and Changes (95%)"
"""

    def __init__(self, api_key, output_dir=None, model_version='o1-2024-12-17', custom_prompt=None):
        """
        Initialize the document classifier
        
        Args:
            api_key (str): OpenAI API key
            output_dir (str, optional): Output directory for results
            model_version (str): Model version to use ('o1-2024-12-17' or 'chatgpt-4o-latest')
            custom_prompt (str, optional): Custom system prompt for this session
        """
        self.client = OpenAI(api_key=api_key)
        self.output_dir = output_dir or os.getcwd()
        self.system_prompt = custom_prompt or self.DEFAULT_PROMPT
        
        # Setup logging first
        self._setup_logging()
        
        self.logger.info(f"Initializing DocumentClassifier with model version: {model_version}")
        self.logger.debug(f"Output directory: {self.output_dir}")
        
        # Validate and set model version
        if model_version not in self.MODELS:
            self.logger.error(f"Invalid model version: {model_version}. Valid options: {', '.join(self.MODELS.keys())}")
            raise ValueError(f"Invalid model version. Choose from: {', '.join(self.MODELS.keys())}")
        
        self.model = self.MODELS[model_version]
        self.model_version = model_version
        self.logger.info(f"Using model: {self.model}")
        
        if custom_prompt:
            self.logger.info("Using custom prompt")
            self.logger.debug(f"Custom prompt: {custom_prompt}")
        
        self._init_categories()
        
    def _setup_logging(self):
        """Set up logging configuration"""
        log_dir = os.path.join(self.output_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'classification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        
        # Create a formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        # File handler with DEBUG level
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Console handler with INFO level
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Set up the logger
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Log initial setup
        self.logger.info("DocumentClassifier logging initialized")
        self.logger.debug(f"Log file created at: {log_file}")

    def _init_categories(self):
        """Initialize document categories and their characteristics"""
        # Main categories with subcategories
        self.document_categories = {
            "Plans & Specifications": [
                "Request for Proposal",
                "Bid Set of Plans and Specifications",
                "Permit Set of Plans",
                "Issued for Construction Plans and Specifications",
                "As-Built Plans and Specifications",
                "Shop Drawings, Submittals (LOGS)",
                "Value Engineering",
                "Plan Reviews",
                "Requests for Information 'RFIs' and LOG"
            ],
            "Key Dates and Schedules": [
                "Notice to Proceed",
                "Notice of Commencement",
                "Temporary Certificate of Occupancy(s) 'TCOs'",
                "Certificate of Occupancy 'CO'",
                "Certificate of Substantial Completion",
                "Approved Baseline Schedule",
                "Schedule Updates"
            ],
            "Contracts and Changes": [
                "Prime Contract Agreement & General Conditions",
                "Contractors Payment & Performance Bond",
                "Change Order Requests 'CORs' and LOG",
                "Change Orders 'COs' and LOG",
                "Subcontractor Contracts",
                "Subcontractor Change Orders and LOG",
                "Backcharges",
                "Construction Change Directives 'CCDs' and LOG"
            ],
            "Meeting Minutes": [
                "Pre-Bid Minutes",
                "Owner Meeting Minutes",
                "Subcontractor Meeting Minutes"
            ],
            "Pay Applications and Job Cost Information": [
                "Engineer's/Owner's Estimate of Project Costs",
                "Contractors Estimate/Bid",
                "Owners Bid Tabulation",
                "Purchase Orders",
                "Monthly Pay Applications and Backup",
                "Contractors Detailed Job Cost Report",
                "Subcontractors Detailed Job Cost Report",
                "Payroll"
            ],
            "Daily Reports / Field Reports": [
                "Owner Representative/Architect",
                "Contractor",
                "Subcontractor"
            ],
            "Inspection Reports and Punchlists": [
                "City/County Inspection Reports",
                "Defective Work Notices",
                "Punchlists",
                "Testing and Certification Documents",
                "Turnover Inspection Report(s)",
                "Threshold Inspection Report",
                "Water Intrusion Report(s)"
            ],
            "Contemporaneous Documentation": [
                "Correspondence",
                "Emails",
                "Memos",
                "Notice of Claims",
                "Transmittals",
                "Photographs"
            ],
            "Miscellaneous": [
                "Movies",
                "Working Estimates",
                "Final Estimate",
                "Bid Summary"
            ]
        }

        # Flatten categories for easier reference
        self.all_categories = [item for sublist in self.document_categories.values() for item in sublist]
        
    def detect_file_type(self, file_path):
        """Detect if file is PDF using python-magic"""
        try:
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(file_path)
            return 'pdf' in file_type.lower()
        except Exception as e:
            self.logger.error(f"Error detecting file type for {file_path}: {str(e)}")
            return False

    def convert_pdf_pages_to_images(self, pdf_path, max_pages=4):
        """Convert first few pages of PDF to base64 encoded images"""
        try:
            # Convert PDF pages to images
            images = convert_from_path(
                pdf_path,
                first_page=1,
                last_page=max_pages,
                dpi=200  # Adjusted DPI for balance of quality and size
            )
            base64_images = []
            
            for i, img in enumerate(images, 1):
                self.logger.debug(f"Converting page {i} of {pdf_path}")
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG', optimize=True)
                img_byte_arr = img_byte_arr.getvalue()
                base64_images.append(base64.b64encode(img_byte_arr).decode('utf-8'))
            
            return base64_images
        except Exception as e:
            self.logger.error(f"Error converting PDF pages to images: {str(e)}")
            return []

    def fix_image_orientation(self, image):
        """Fix rotation and skew in the image"""
        # Convert PIL Image to OpenCV format
        cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Get orientation info from Tesseract
        try:
            osd = pytesseract.image_to_osd(cv_image)
            angle = int(osd.split('\nRotate: ')[1].split('\n')[0])
            
            # Rotate if needed
            if angle != 0:
                (h, w) = cv_image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                cv_image = cv2.warpAffine(cv_image, M, (w, h), 
                                        flags=cv2.INTER_CUBIC, 
                                        borderMode=cv2.BORDER_REPLICATE)
        except:
            # If Tesseract fails, try to detect skew using contours
            self.logger.debug("Tesseract OSD failed, attempting contour-based deskew")
            
        # Convert to grayscale for skew detection
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # Threshold the image
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Detect skew angle
        coords = np.column_stack(np.where(thresh > 0))
        angle = cv2.minAreaRect(coords)[-1]
        
        # The angle is between -90 and 0 degrees
        if angle < -45:
            angle = 90 + angle
            
        # Rotate the image to deskew it if skew is significant
        if abs(angle) > 0.5:
            (h, w) = cv_image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            cv_image = cv2.warpAffine(cv_image, M, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
        
        # Convert back to PIL Image
        return Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))

    def preprocess_image(self, image):
        """Preprocess the image for better classification"""
        self.logger.debug("Starting image preprocessing")
        
        # Fix orientation
        self.logger.debug("Fixing image orientation...")
        try:
            image = self.fix_image_orientation(image)
            self.logger.debug("Orientation correction complete")
        except Exception as e:
            self.logger.warning(f"Orientation correction failed: {str(e)}", exc_info=True)
            self.logger.debug("Continuing with original orientation")
        
        # Image enhancement
        self.logger.debug("Enhancing image quality...")
        try:
            # Convert to numpy array for OpenCV operations
            cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            
            # Enhance contrast
            self.logger.debug("Applying CLAHE contrast enhancement...")
            lab = cv2.cvtColor(cv_image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            enhanced = cv2.merge((cl,a,b))
            enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)
            
            # Denoise
            self.logger.debug("Applying denoising...")
            denoised = cv2.fastNlMeansDenoisingColored(enhanced)
            
            # Convert back to PIL Image
            final_image = Image.fromarray(cv2.cvtColor(denoised, cv2.COLOR_BGR2RGB))
            self.logger.debug("Image enhancement complete")
            return final_image
            
        except Exception as e:
            self.logger.error(f"Image enhancement failed: {str(e)}", exc_info=True)
            self.logger.warning("Returning original image without enhancement")
            return image

    def classify_document(self, file_path: str):
        """Classify a document using selected OpenAI model"""
        self.logger.info(f"Starting classification of: {file_path}")
        
        try:
            # Skip non-PDF files
            if not file_path.lower().endswith('.pdf'):
                self.logger.warning(f"Skipping non-PDF file: {file_path}")
                return "Unsupported File", 0

            # Convert first page of PDF to image
            self.logger.debug("Converting PDF to image...")
            try:
                images = convert_from_path(file_path, first_page=1, last_page=1)
                if not images:
                    self.logger.error("PDF conversion returned no images")
                    raise ValueError("Failed to convert PDF to image")
                image = images[0]
                self.logger.debug(f"PDF converted successfully. Image size: {image.size}")
                
                # Preprocess the image
                self.logger.debug("Starting image preprocessing...")
                image = self.preprocess_image(image)
                self.logger.debug("Image preprocessing completed")
                
            except Exception as e:
                self.logger.error(f"Error converting/preprocessing PDF: {str(e)}", exc_info=True)
                raise

            # Convert image to base64
            self.logger.debug("Converting image to base64...")
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format='JPEG', quality=95)
            img_byte_arr = img_byte_arr.getvalue()
            base64_image = base64.b64encode(img_byte_arr).decode('utf-8')
            self.logger.debug(f"Base64 conversion complete. Size: {len(base64_image)} chars")

            # Create the API message
            self.logger.debug("Preparing API request...")
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Please classify this document according to the categories above."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}",
                                "detail": "high"  # Request high detail analysis
                            }
                        }
                    ]
                }
            ]

            # Log the full API request details
            self.logger.debug("API Request Details:")
            self.logger.debug(f"Model: {self.model}")
            self.logger.debug(f"System Prompt: {self.system_prompt}")
            self.logger.debug(f"Messages Structure: {messages}")
            self.logger.debug(f"Image size (bytes): {len(base64_image)}")

            # Make API call with simplified parameters
            self.logger.info("Sending request to OpenAI API...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )

            # Log the complete response
            self.logger.debug(f"Complete API Response: {response}")
            
            # Parse the response
            result = response.choices[0].message.content.strip()
            self.logger.info(f"Received classification result: {result}")
            
            if not result:
                self.logger.error("Received empty response from API")
                raise ValueError("Empty response from API")
            
            # Extract category number and confidence
            self.logger.debug(f"Parsing response: {result}")
            import re
            match = re.match(r'(\d+)\.\s+([^(]+)\s*\((\d+)%\)', result)
            if not match:
                self.logger.error(f"Failed to parse response format: {result}")
                raise ValueError(f"Unexpected response format: {result}")
            
            category_num, category_name, confidence = match.groups()
            self.logger.info(f"Successfully classified as: {category_name} ({confidence}%)")
            return category_name.strip(), float(confidence)

        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}", exc_info=True)
            raise

    def process_folder(self, folder_path, batch_size=10):
        """Process all PDF documents in a folder with batch processing"""
        results = []
        processed_count = 0
        error_count = 0
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.logger.info(f"Starting batch processing of folder: {folder_path}")
        
        # Get list of all files
        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
        total_files = len(files)
        
        for file_name in files:
            processed_count += 1
            file_path = os.path.join(folder_path, file_name)
            
            self.logger.info(f"Processing file {processed_count}/{total_files}: {file_name}")
            
            try:
                doc_type, confidence = self.classify_document(file_path)
                results.append({
                    'Filename': file_name,
                    'Document Type': doc_type,
                    'Confidence Score': f"{confidence:.1f}%"
                })
                
                # Save intermediate results after each batch
                if processed_count % batch_size == 0:
                    self._save_results(results)
                    
            except Exception as e:
                error_count += 1
                self.logger.error(f"Error processing {file_name}: {str(e)}")
                results.append({
                    'Filename': file_name,
                    'Document Type': 'Processing Error',
                    'Confidence Score': '0.0%'
                })

        # Save final results
        output_df = self._save_results(results)
        
        # Log summary
        self.logger.info(f"""
Processing Summary:
-----------------
Total Files: {total_files}
Successfully Processed: {processed_count - error_count}
Errors: {error_count}
Success Rate: {((processed_count - error_count) / total_files * 100):.1f}%
""")
        
        return output_df

    def _save_results(self, results):
        """Save results to Excel file"""
        df = pd.DataFrame(results)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_dir, f'document_classification_results_{timestamp}.xlsx')
        
        df.to_excel(output_path, index=False)
        self.logger.info(f"Results saved to: {output_path}")
        
        return df

def main():
    """Main function to run the document classifier"""
    try:
        # Get API key from environment variable
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Please set the OPENAI_API_KEY environment variable")

        # Get output directory
        output_dir = input("Enter output directory path (press Enter for current directory): ").strip()
        if not output_dir:
            output_dir = os.getcwd()
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Get model version from user
        model_version = input("Enter model version (press Enter for default 'o1-2024-12-17'): ").strip()
        model_version = model_version if model_version in DocumentClassifier.MODELS else 'o1-2024-12-17'

        # Initialize classifier
        classifier = DocumentClassifier(api_key, output_dir, model_version)

        # Get folder path from user
        folder_path = input("Enter the folder path to process: ").strip()
        if not os.path.exists(folder_path):
            raise ValueError("Folder path does not exist")

        # Get batch size from user
        batch_size = input("Enter batch size (press Enter for default 10): ").strip()
        batch_size = int(batch_size) if batch_size else 10

        # Process folder and display results
        results_df = classifier.process_folder(folder_path, batch_size)
        print("\nClassification Results Summary:")
        print(results_df.groupby('Document Type').size())

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())
    