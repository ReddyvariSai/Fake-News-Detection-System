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
    
    if file_size < 100:
        print("⚠️ Warning: File seems too small. Might be empty or corrupted.")
    
    try:
        # Try to load the vectorizer
        vectorizer = joblib.load(filepath)
        print(f"✅ Vectorizer loaded successfully!")
        print(f"   Type: {type(vectorizer).__name__}")
        
        # Check common attributes
        if hasattr(vectorizer, 'get_feature_names_out'):
            features = vectorizer.get_feature_names_out()
            print(f"✅ Has get_feature_names_out(): Yes")
            print(f"   Vocabulary size: {len(features)}")
            if len(features) > 0:
                print(f"   Sample features: {features[:5]}")
        
        if hasattr(vectorizer, 'transform'):
            print(f"✅ Has transform() method: Yes")
            
            # Test transformation
            try:
                test_texts = ["test text for verification"]
                transformed = vectorizer.transform(test_texts)
                print(f"✅ Transform works: shape {transformed.shape}")
            except Exception as e:
                print(f"⚠️ Transform test failed: {e}")
        
        if hasattr(vectorizer, 'metadata'):
            print(f"✅ Has metadata: Yes")
            print(f"   Created: {vectorizer.metadata.get('created_date', 'unknown')}")
            print(f"   Version: {vectorizer.metadata.get('version', 'unknown')}")
        
        # Check if it's a TF-IDF vectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer
        if isinstance(vectorizer, TfidfVectorizer):
            print(f"✅ Type: TF-IDF Vectorizer")
            print(f"   Max features: {vectorizer.max_features}")
            print(f"   N-gram range: {vectorizer.ngram_range}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading vectorizer: {e}")
        print("\n💡 The file might be corrupted. Try creating a new one:")
        print("   python create_vectorizer.py")
        return False

def create_minimal_vectorizer():
    """Create a minimal vectorizer if none exists"""
    
    if not os.path.exists('models/vectorizer.pkl'):
        print("\n📦 Creating minimal vectorizer...")
        
        import joblib
        from sklearn.feature_extraction.text import TfidfVectorizer
        import os
        
        # Create minimal vectorizer
        vectorizer = TfidfVectorizer(max_features=100)
        
        # Fit with sample data
        sample_texts = [
            "real news article",
            "fake news breaking",
            "government report",
            "conspiracy theory"
        ]
        vectorizer.fit(sample_texts)
        
        # Save
        os.makedirs('models', exist_ok=True)
        joblib.dump(vectorizer, 'models/vectorizer.pkl')
        
        print("✅ Minimal vectorizer created!")
        return True
    return False

if __name__ == "__main__":
    print("=" * 60)
    print("VECTORIZER.PKL VERIFICATION")
    print("=" * 60)
    
    # Try to create if missing
    if not os.path.exists('models/vectorizer.pkl'):
        create_minimal_vectorizer()
    
    # Verify
    success = verify_vectorizer()
    
    if success:
        print("\n🎉 Vectorizer is ready to use!")
        print("\n📝 Example usage:")
        print("""
import joblib

# Load the vectorizer
vectorizer = joblib.load('models/vectorizer.pkl')

# Transform text
texts = ["Your news article here"]
features = vectorizer.transform(texts)

print(f"Features shape: {features.shape}")
print(f"Vocabulary size: {len(vectorizer.get_feature_names_out())}")
        """)
    else:
        print("\n❌ Vectorizer verification failed.")
        print("   Try creating a new vectorizer with:")
        print("   python create_vectorizer.py")
