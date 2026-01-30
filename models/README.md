
models/
├── __init__.py
├── model_manager.py
├── current/                          # Symlink or reference to current version
│   ├── fake_news_model.pkl
│   ├── vectorizer.pkl
│   ├── preprocessor.pkl
│   ├── scaler.pkl
│   └── metadata.json
├── versions/                         # All historical versions
│   ├── v1.0.0/
│   │   ├── model.pkl
│   │   ├── vectorizer.pkl
│   │   ├── preprocessor.pkl
│   │   ├── scaler.pkl
│   │   ├── metadata.json
│   │   └── metrics.json            # Add performance metrics
│   └── v1.1.0/
│       └── ...
├── candidates/                       # Models under evaluation
│   ├── random_forest/
│   ├── xgboost/
│   └── ensemble/
├── artifacts/                        # Training artifacts
│   ├── feature_importance.png
│   ├── confusion_matrix.png
│   └── training_logs.csv
└── configs/                          # Model configurations
    ├── base_config.yaml
    ├── rf_config.yaml
    └── xgb_config.yaml
