import os
from pathlib import Path

def create_folder_structure(base_path: Path):
    """Create the folder structure for document classification testing."""
    
    # Main folders structure
    folders = {
        'input': {  # Folder for documents to be classified
            'pdfs': {},
            'test_files': {},
        },
        'output': {  # Folder for classification results
            'reports': {},
            'logs': {},
        },
        'samples': {  # Sample documents organized by category
            '01_plans_and_specifications': {
                'request_for_proposal': {},
                'bid_sets': {},
                'permit_sets': {},
                'shop_drawings': {},
                'rfis': {},
            },
            '02_key_dates_and_schedules': {
                'notices': {},
                'certificates': {},
                'schedules': {},
            },
            '03_contracts_and_changes': {
                'prime_contracts': {},
                'change_orders': {},
                'subcontracts': {},
            },
            '04_meeting_minutes': {
                'pre_bid': {},
                'owner': {},
                'subcontractor': {},
            },
            '05_pay_applications': {
                'estimates': {},
                'pay_apps': {},
                'job_costs': {},
            },
            '06_daily_reports': {
                'owner': {},
                'contractor': {},
                'subcontractor': {},
            },
            '07_inspection_reports': {
                'city_county': {},
                'punchlists': {},
                'testing': {},
            },
            '08_documentation': {
                'correspondence': {},
                'emails': {},
                'notices': {},
            },
            '09_miscellaneous': {
                'photos': {},
                'estimates': {},
            }
        },
        'config': {},  # Configuration files
        'temp': {},    # Temporary processing files
    }

    def create_folders(path, structure):
        for folder, subfolders in structure.items():
            folder_path = os.path.join(path, folder)
            os.makedirs(folder_path, exist_ok=True)
            if subfolders:
                create_folders(folder_path, subfolders)

    # Create the folder structure
    create_folders(base_path, folders)
    
    # Create a README file in each sample category folder
    for category in folders['samples'].keys():
        readme_path = os.path.join(base_path, 'samples', category, 'README.md')
        with open(readme_path, 'w') as f:
            category_name = category[3:].replace('_', ' ').title()
            f.write(f"""# {category_name}

Place sample documents for {category_name} in appropriate subfolders.

## Subfolders:
""")
            # List subfolders
            for subfolder in folders['samples'][category].keys():
                f.write(f"- {subfolder.replace('_', ' ').title()}\n")

    # Create README files in main directories
    readmes = {
        'input/pdfs': "Place PDF files here for classification",
        'input/test_files': "Place test files of various formats here",
        'output/reports': "Classification results will be saved here",
        'output/logs': "System logs will be stored here",
        'config': "Configuration files",
        'temp': "Temporary processing files"
    }

    for folder, description in readmes.items():
        readme_path = os.path.join(base_path, folder, 'README.md')
        with open(readme_path, 'w') as f:
            f.write(f"# {folder.split('/')[-1].title()}\n\n{description}")

    return True

if __name__ == "__main__":
    base_path = Path("A:/doctype")
    if create_folder_structure(base_path):
        print(f"Folder structure created successfully at {base_path}")
    else:
        print("Error creating folder structure") 