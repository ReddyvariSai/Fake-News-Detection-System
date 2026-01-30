```

data/
├── __init__.py
├── data_manager.py                    # Data loading and management
├── data_processor.py                  # Data processing pipeline
├── data_validator.py                  # Data validation and quality checks
├── data_splitter.py                   # Train/test/validation splitting
├── data_augmenter.py                  # Data augmentation techniques
├── README.md                          # Data documentation
├── raw/                               # Raw data (never modify)
│   ├── fake_news.csv                  # Main dataset
│   ├── kaggle_fake_news/              # External datasets
│   │   ├── train.csv
│   │   └── test.csv
│   ├── liar_dataset/                  # LIAR dataset
│   │   ├── train.tsv
│   │   ├── test.tsv
│   │   └── valid.tsv
│   ├── twitter_fake_news/             # Twitter datasets
│   │   └── tweets.csv
│   └── README.md                      # Raw data documentation
├── processed/                         # Processed data
│   ├── cleaned_news.csv               # Cleaned dataset
│   ├── train/
│   │   ├── train.csv                  # Training set
│   │   ├── train_features.npy         # Processed features
│   │   ├── train_labels.npy           # Processed labels
│   │   └── train_metadata.json        # Training metadata
│   ├── test/
│   │   ├── test.csv                   # Test set
│   │   ├── test_features.npy
│   │   ├── test_labels.npy
│   │   └── test_metadata.json
│   ├── validation/
│   │   ├── validation.csv
│   │   ├── validation_features.npy
│   │   ├── validation_labels.npy
│   │   └── validation_metadata.json
│   ├── features/
│   │   ├── vocabulary.pkl             # Feature vocabulary
│   │   ├── vectorizer.pkl             # Fitted vectorizer
│   │   ├── scaler.pkl                 # Fitted scaler
│   │   └── feature_names.json         # Feature names
│   └── statistics/                    # Data statistics
│       ├── data_statistics.json
│       ├── class_distribution.png
│       └── preprocessing_report.md
├── external/                          # External data sources
│   ├── sentiment_lexicons/            # Sentiment dictionaries
│   │   ├── positive_words.txt
│   │   └── negative_words.txt
│   ├── domain_lists/                  # Domain classification
│   │   ├── credible_domains.txt
│   │   └── questionable_domains.txt
│   └── embeddings/                    # Pre-trained embeddings
│       ├── glove.6B.100d.txt
│       └── fasttext.pkl
└── cache/                             # Intermediate cache
    ├── temporary_features/
    └── processed_chunks/

```
