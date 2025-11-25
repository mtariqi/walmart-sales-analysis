# Create a simple installation script
#!/usr/bin/env python3
"""
Installation script for Walmart Sales Analysis
Automatically handles different Python versions
"""

import sys
import subprocess
import platform

def run_command(cmd):
    """Run a shell command and return success status"""
    try:
        subprocess.check_call(cmd, shell=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running: {cmd}")
        print(f"Error: {e}")
        return False

def main():
    python_version = sys.version_info
    system = platform.system()
    
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    print(f"System: {system}")
    print("Installing dependencies...")
    
    # Upgrade pip first
    print("\n1. Upgrading pip...")
    run_command("pip install --upgrade pip")
    
    # Install core data science packages
    print("\n2. Installing core packages...")
    core_packages = [
        "numpy",
        "pandas", 
        "scipy",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "jupyter",
        "jupyterlab"
    ]
    
    for package in core_packages:
        print(f"   Installing {package}...")
        if not run_command(f"pip install {package}"):
            print(f"   Failed to install {package}, trying with --pre")
            run_command(f"pip install --pre {package}")
    
    # Install utility packages
    print("\n3. Installing utility packages...")
    utility_packages = [
        "python-dotenv",
        "tqdm", 
        "joblib",
        "openpyxl"
    ]
    
    for package in utility_packages:
        print(f"   Installing {package}...")
        run_command(f"pip install {package}")
    
    print("\n4. Installing development tools...")
    dev_packages = [
        "pytest",
        "black",
        "flake8"
    ]
    
    for package in dev_packages:
        print(f"   Installing {package}...")
        run_command(f"pip install {package}")
    
    print("\nInstallation complete!")
    print("Testing imports...")
    
    # Test imports
    test_imports = [
        "import numpy",
        "import pandas",
        "import sklearn",
        "import matplotlib",
        "import seaborn"
    ]
    
    for import_stmt in test_imports:
        try:
            exec(import_stmt)
            print(f"✓ {import_stmt} - SUCCESS")
        except ImportError as e:
            print(f"✗ {import_stmt} - FAILED: {e}")

if __name__ == "__main__":
    main()
