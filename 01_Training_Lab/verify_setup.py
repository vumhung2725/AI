"""
Quick Test Script - Verify everything is working
Run this to check if your environment is set up correctly
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check if all required packages are installed"""
    print("🔍 Checking dependencies...")
    
    dependencies = {
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy',
        'tensorflow': 'tensorflow',
        'streamlit': 'streamlit',
        'sklearn': 'scikit-learn'
    }
    
    missing = []
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"   ✅ {module}")
        except ImportError:
            print(f"   ❌ {module} (install: pip install {package})")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print(f"Run: pip install {' '.join(missing)}")
        return False
    else:
        print(f"\n✅ All dependencies installed!")
        return True


def check_structure():
    """Check folder structure"""
    print("\n🔍 Checking project structure...")
    
    required_dirs = [
        '01_Training_Lab/data_collection',
        '01_Training_Lab/model_training',
        '01_Training_Lab/utils',
        '02_Web_Client/js',
        '02_Web_Client/assets/models',
        '03_Docs_QC/reports',
    ]
    
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"   ✅ {dir_path}/")
        else:
            print(f"   ⚠️  Missing: {dir_path}/")
    
    print("✅ Structure check complete!")


def check_files():
    """Check key files exist"""
    print("\n🔍 Checking key files...")
    
    files = {
        '01_Training_Lab/data_collection/1_collect_seq.py': 'Data collection script',
        '01_Training_Lab/model_training/2_train_hybrid.py': 'Training script',
        '01_Training_Lab/model_training/3_convert_tfjs.py': 'TFJS conversion script',
        '01_Training_Lab/utils/data_utils.py': 'Utility functions',
        'main.py': 'Streamlit app',
        'requirements.txt': 'Dependencies list',
    }
    
    for file_path, description in files.items():
        path = Path(file_path)
        if path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ Missing: {file_path}")
    
    print("✅ Files check complete!")


def print_next_steps():
    """Print usage instructions"""
    print("\n" + "="*60)
    print("🚀 NEXT STEPS")
    print("="*60)
    print("""
1. COLLECT DATA:
   python 01_Training_Lab/data_collection/1_collect_seq.py \\
       --action_name="hello" \\
       --num_samples=30
   
   (Repeat for each action you want to train)

2. TRAIN MODEL:
   python 01_Training_Lab/model_training/2_train_hybrid.py

3. RUN APP:
   streamlit run main.py

4. (OPTIONAL) EXPORT TO BROWSER:
   python 01_Training_Lab/model_training/3_convert_tfjs.py

📖 For detailed instructions, see: 01_Training_Lab/QUICKSTART.md
""")
    print("="*60)


def main():
    print("\n" + "="*60)
    print("✨ Sign Language Recognition - Setup Verification")
    print("="*60 + "\n")
    
    # Run checks
    deps_ok = check_dependencies()
    check_structure()
    check_files()
    
    if deps_ok:
        print("\n✅ Setup looks good! You're ready to go.")
        print_next_steps()
    else:
        print("\n❌ Please install missing dependencies first.")
        print("Run: pip install -r requirements.txt")
    
    print()


if __name__ == "__main__":
    main()
