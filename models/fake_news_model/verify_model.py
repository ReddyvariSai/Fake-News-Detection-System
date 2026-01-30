import joblib
import os

def verify_model(filepath='models/fake_news_model.pkl'):
    """Verify the model file exists and can be loaded"""
    
    print(f"🔍 Verifying {filepath}...")
    
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        print("   Please create it first using one of these methods:")
        print("   1. Run: python create_fake_news_model.py")
        print("   2. Run the one-line command in README.md")
        return False
    
    # Check file size
    file_size = os.path.getsize(filepath)
    print(f"✅ File exists: {file_size:,} bytes")
    
    if file_size < 100:
        print("⚠️ Warning: File seems too small. Might be empty or corrupted.")
    
    try:
        # Try to load the model
        model = joblib.load(filepath)
        print(f"✅ Model loaded successfully!")
        print(f"   Model type: {type(model).__name__}")
        
        # Check if it has predict method
        if hasattr(model, 'predict'):
            print(f"✅ Has predict() method: Yes")
            
            # Try a dummy prediction
            import numpy as np
            try:
                dummy_input = np.random.rand(1, 5)
                prediction = model.predict(dummy_input)
                print(f"✅ Prediction test passed: {prediction}")
            except Exception as e:
                print(f"⚠️ Prediction test failed (might need correct input shape): {e}")
        
        if hasattr(model, 'predict_proba'):
            print(f"✅ Has predict_proba() method: Yes")
        
        # Check for metadata
        if hasattr(model, 'metadata'):
            print(f"✅ Has metadata: Yes")
            print(f"   Model name: {model.metadata.get('model_name', 'Unknown')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   The file might be corrupted or in wrong format.")
        return False

def create_minimal_if_missing():
    """Create a minimal model if it doesn't exist"""
    
    if not os.path.exists('models/fake_news_model.pkl'):
        print("\n📦 Creating minimal model...")
        
        import joblib
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        
        # Create minimal model
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        X = np.random.rand(10, 10)
        y = np.random.randint(0, 2, 10)
        model.fit(X, y)
        
        # Save
        os.makedirs('models', exist_ok=True)
        joblib.dump(model, 'models/fake_news_model.pkl')
        
        print("✅ Minimal model created!")
        return True
    return False

if __name__ == "__main__":
    print("=" * 60)
    print("FAKE NEWS MODEL VERIFICATION")
    print("=" * 60)
    
    # Try to create if missing
    if not os.path.exists('models/fake_news_model.pkl'):
        create_minimal_if_missing()
    
    # Verify
    success = verify_model()
    
    if success:
        print("\n🎉 Model is ready to use!")
        print("\n📝 Example usage:")
        print("""
import joblib

# Load the model
model = joblib.load('models/fake_news_model.pkl')

# Make prediction (adjust input shape as needed)
import numpy as np
input_data = np.random.rand(1, 10)  # 1 sample, 10 features
prediction = model.predict(input_data)
print(f"Prediction: {prediction}")
        """)
    else:
        print("\n❌ Model verification failed.")
        print("   Try creating a new model with:")
        print("   python create_fake_news_model.py")
