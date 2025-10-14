#!/usr/bin/env python3
"""
Setup script for Time Series Forecasting Project
"""

import os
import sys
import subprocess
import platform

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required.")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python version: {sys.version.split()[0]}")

def install_requirements():
    """Install required packages from requirements.txt."""
    print("Installing required packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ All packages installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        sys.exit(1)

def create_directories():
    """Create necessary directories."""
    directories = [
        "data",
        "results",
        "results/plots",
        "results/models"
    ]
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory exists: {directory}")

def check_data_files():
    """Check if required data files exist."""
    required_files = [
        "data/UK_Weekly_Trended_Timeline_from_200101_202429.xlsx",
        "data/ISBN_List.xlsx"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
        else:
            print(f"✓ Data file found: {file_path}")
    
    if missing_files:
        print("\n⚠️  Missing data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nPlease download the required data files and place them in the data/ directory.")
        print("See data/README.md for more information.")
        return False
    
    return True

def main():
    """Main setup function."""
    print("Time Series Forecasting Project Setup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Install requirements
    install_requirements()
    
    # Check data files
    data_available = check_data_files()
    
    print("\n" + "=" * 40)
    if data_available:
        print("✓ Setup completed successfully!")
    print("\nYou can now run the analysis with:")
    print("  python time_series_forecasting_analysis.py")
    else:
        print("⚠️  Setup completed with warnings.")
        print("Please add the required data files before running the analysis.")
    
    print("\nFor more information, see README.md")

if __name__ == "__main__":
    main()
