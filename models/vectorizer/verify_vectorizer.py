import os
import joblib

def verify_vectorizer(filepath='models/vectorizer.pkl'):
    """Verify the vectorizer.pkl file"""
    
    print("🔍 Verifying vectorizer.pkl...")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        print("\n💡 To create it, run one of these:")
        print("   1. python create_vectorizer.py")
        print("   2. Use the one-line command in terminal")
        return False
    
    # Check file size
    file_size = os.path.getsize(filepath)
    print(f"✅ File exists: {file_size:,} bytes")
