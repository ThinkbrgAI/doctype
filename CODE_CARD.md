# Construction Document Classifier

## Overview
AI-powered document classification system for construction project documents. Uses GPT-4 Vision models to analyze and categorize documents based on industry-standard categories and subcategories.

## Key Features
- Advanced document classification with 9 main categories and detailed subcategories
- Recognition of common construction document elements and characteristics
- High-accuracy classification using GPT-4 Vision models
- Detailed confidence scoring for both category and subcategory matches
- Image preprocessing for skewed or rotated documents
- Comprehensive logging system
- Multi-sheet Excel export with summaries

## Document Categories
1. Plans & Specifications
   - Recognizes technical drawings, blueprints, specifications
   - Identifies standard title blocks, stamps, drawing numbers
   - Detects revision marks, grid lines, detail callouts

2. Key Dates and Schedules
   - Identifies timeline information, milestone dates
   - Recognizes schedule formats and critical path indicators
   - Detects official stamps and recording information

[Continue with other categories...]

## Usage
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the application:
   ```bash
   python run_classifier.py
   ```

3. For command-line interface:
   ```bash
   python run_classifier.py --cli
   ```

## Configuration
- Secure API key storage
- Customizable system prompts
- Support for multiple GPT-4 Vision models
- Configurable batch processing

## Output
- Detailed classification results with confidence scores
- Category and subcategory identification
- Excel reports with multiple summary sheets
- Comprehensive processing logs

## Models Supported
- o1-2024-12-17 (GPT-4 Vision specific version)
- chatgpt-4o-latest (GPT-4 Turbo)

## Requirements
- Python 3.8+
- OpenAI API key
- PDF processing capabilities
- Excel support for reporting 