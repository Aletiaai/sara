# Create this as debug_env.py in your sara/ directory
# Run with: python debug_env.py

import os
from pathlib import Path
from dotenv import load_dotenv

print("=== DEBUGGING .env FILE ===")

# Find the project root and load the .env file
current_path = Path.cwd()
ENV_PATH = current_path / ".env"

print(f"Current working directory: {current_path}")
print(f"Looking for .env file at: {ENV_PATH}")
print(f".env file exists: {ENV_PATH.exists()}")

if ENV_PATH.exists():
    # Read the raw file content
    with open(ENV_PATH, 'r') as f:
        content = f.read()
    print(f"\nRaw .env file content:")
    print("=" * 50)
    print(content)
    print("=" * 50)
    
    # Load with dotenv
    load_dotenv(dotenv_path=ENV_PATH)
    
    # Check what was loaded
    google_creds = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    print(f"\nLoaded GOOGLE_APPLICATION_CREDENTIALS: '{google_creds}'")
    
    # Check if the file exists
    if google_creds:
        key_file_path = current_path / google_creds
        print(f"Looking for key file at: {key_file_path}")
        print(f"Key file exists: {key_file_path.exists()}")
        
        # List files in current directory
        print(f"\nFiles in current directory:")
        for file in current_path.glob("*.json"):
            print(f"  - {file.name}")
else:
    print("ERROR: .env file not found!")