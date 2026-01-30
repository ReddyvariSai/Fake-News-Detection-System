python -c "
import joblib, numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

# Create directory
os.makedirs('models', exist_ok=True)

# Create model
model = RandomForestClassifier(n_estimators=10, random_state=42)
X = np.random.rand(10, 5)
y = np.random.randint(0, 2, 10)
model.fit(X, y)

# Save
joblib.dump(model, 'models/fake_news_model.pkl')
print('✅ fake_news_model.pkl created! Size:', os.path.getsize('models/fake_news_model.pkl'), 'bytes')
"
