#!/usr/bin/env python3
"""
Setup script for Disease Diagnosis Screening Project
Adapted for existing folder structure
"""

import subprocess
import sys
import os
import importlib

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required.")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        print("   Please update Python and try again.")
        return False
    else:
        print(f"✅ Python version {version.major}.{version.minor}.{version.micro} is compatible")
        return True

def install_requirements():
    """Install required packages"""
    print("\n📦 Installing required packages...")
    
    requirements = [
        'pandas>=1.5.0',
        'numpy>=1.21.0',
        'scikit-learn>=1.2.0',
        'tensorflow>=2.10.0',
        'matplotlib>=3.5.0',
        'seaborn>=0.11.0',
        'requests>=2.28.0',
        'scikeras>=0.9.0'
    ]
    
    for package in requirements:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✅ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    return True

def test_imports():
    """Test if all required modules can be imported"""
    print("\n🧪 Testing module imports...")
    
    modules_to_test = [
        ('pandas', 'pd'),
        ('numpy', 'np'),
        ('sklearn', None),
        ('tensorflow', 'tf'),
        ('matplotlib.pyplot', 'plt'),
        ('seaborn', 'sns'),
        ('requests', None),
        ('scikeras', None)
    ]
    
    all_imports_successful = True
    
    for module_name, alias in modules_to_test:
        try:
            if alias:
                exec(f"import {module_name} as {alias}")
            else:
                exec(f"import {module_name}")
            print(f"✅ {module_name} imported successfully")
        except ImportError as e:
            print(f"❌ Failed to import {module_name}: {e}")
            all_imports_successful = False
    
    return all_imports_successful

def check_file_structure():
    """Check if required files and folders are present"""
    print("\n📁 Checking project file structure...")
    
    # Check for main folders
    required_folders = [
        'diabetes-onset-detection',
        'heart-disease-diagnosis', 
        'autism-screening',
        'dataset'
    ]
    
    # Find folders that match (allowing for partial names)
    found_folders = []
    for item in os.listdir('.'):
        if os.path.isdir(item):
            for req_folder in required_folders:
                if req_folder.replace('-', '').replace('_', '') in item.replace('-', '').replace('_', ''):
                    found_folders.append(item)
                    print(f"✅ Found folder: {item}")
                    break
    
    # Check for dataset
    if os.path.exists('dataset'):
        print("✅ dataset/ folder found")
        if os.path.exists('dataset/autism-data.csv'):
            print("✅ autism-data.csv found in dataset/")
        else:
            print("ℹ️  autism-data.csv not found (will create sample data)")
    
    # Check for existing Python files
    python_files_found = 0
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.py') and file != 'setup.py':
                python_files_found += 1
                print(f"✅ Found Python file: {os.path.join(root, file)}")
    
    if python_files_found == 0:
        print("⚠️  No existing Python files found")
        print("   Please add the updated Python files to their respective folders")
        return False
    
    print(f"✅ Found {python_files_found} Python files")
    return True

def create_folders_if_needed():
    """Create any missing folders"""
    print("\n📂 Creating any missing folders...")
    
    folders_to_create = [
        'breast-cancer',
        'models',
        'results',
        'results/charts',
        'results/reports'
    ]
    
    for folder in folders_to_create:
        if not os.path.exists(folder):
            try:
                os.makedirs(folder, exist_ok=True)
                print(f"✅ Created folder: {folder}")
            except Exception as e:
                print(f"❌ Failed to create {folder}: {e}")

def create_test_run():
    """Create a simple test to verify everything works"""
    print("\n🚀 Creating test verification...")
    
    test_code = '''
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Create simple test data
np.random.seed(42)
X = np.random.randn(100, 4)
y = (X.sum(axis=1) > 0).astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create simple model
model = Sequential([
    Dense(8, input_dim=4, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print("✅ All components working correctly!")
print("🎉 Setup verification completed successfully!")
'''
    
    try:
        exec(test_code)
        return True
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def show_usage_instructions():
    """Show how to use the updated files"""
    print("\n📚 How to run your updated programs:")
    print("-" * 50)
    
    # Find actual folder names
    folders = [d for d in os.listdir('.') if os.path.isdir(d) and d not in ['.git', '__pycache__', 'models', 'results', 'dataset']]
    
    for folder in folders:
        if 'diabetes' in folder.lower():
            print(f"🍬 Diabetes Detection:")
            print(f"   cd {folder}")
            print(f"   python diabetes_updated.py")
            print()
        elif 'heart' in folder.lower():
            print(f"❤️  Heart Disease Diagnosis:")
            print(f"   cd {folder}")
            print(f"   python heart-disease_updated.py")
            print()
        elif 'autism' in folder.lower():
            print(f"🧠 Autism Screening:")
            print(f"   cd {folder}")
            print(f"   python autism_updated.py")
            print()
        elif 'breast' in folder.lower() or 'cancer' in folder.lower():
            print(f"🔬 Breast Cancer Detection:")
            print(f"   cd {folder}")
            print(f"   python breast-cancer_updated.py")
            print()

def main():
    """Main setup function"""
    print("🏥 Disease Diagnosis Screening Project Setup")
    print("Adapted for your existing folder structure")
    print("=" * 60)
    
    # Step 1: Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Step 2: Check file structure
    if not check_file_structure():
        print("\n⚠️  Please add the updated Python files to their respective folders.")
        print("   Check the instructions for which files go where.")
    
    # Step 3: Create missing folders
    create_folders_if_needed()
    
    # Step 4: Install requirements
    if not install_requirements():
        print("\n❌ Package installation failed. Please check your internet connection and try again.")
        sys.exit(1)
    
    # Step 5: Test imports
    if not test_imports():
        print("\n❌ Module import test failed. Please check the error messages above.")
        sys.exit(1)
    
    # Step 6: Run verification test
    if not create_test_run():
        print("\n❌ Setup verification failed. Please check the error messages above.")
        sys.exit(1)
    
    # Success message
    print("\n" + "=" * 60)
    print("🎉 SETUP COMPLETED SUCCESSFULLY! 🎉")
    print("=" * 60)
    
    show_usage_instructions()
    
    print("📚 For detailed help, see your existing README.md")
    print()
    print("⚠️  Remember: These tools are for educational purposes only!")
    print("   Always consult healthcare professionals for medical advice.")

if __name__ == "__main__":
    main()