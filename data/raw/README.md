# Raw Data Directory

This directory contains all raw data files. **Never modify files in this directory directly.**

## Dataset Sources

### 1. fake_news.csv
**Description**: Primary fake news detection dataset
**Source**: Internal collection or primary source
**Format**: CSV
**Columns**:
- `text`: News article text content
- `title`: Article title
- `label`: Binary label (0=real, 1=fake)
- `source`: News source/domain
- `date`: Publication date
- `author`: Article author (if available)

### 2. kaggle_fake_news/
**Description**: Kaggle Fake News Detection Challenge dataset
**Source**: https://www.kaggle.com/c/fake-news
**Files**:
- `train.csv`: Training data
- `test.csv`: Test data (without labels)

### 3. liar_dataset/
**Description**: LIAR dataset for fake news detection
**Source**: https://www.cs.ucsb.edu/~william/data/liar_dataset.zip
**Files**:
- `train.tsv`: Training split
- `test.tsv`: Test split
- `valid.tsv`: Validation split

**Columns**:
- Column 1: ID
- Column 2: Label (true, mostly-true, half-true, barely-true, false, pants-fire)
- Column 3: Statement
- Columns 4-13: Various metadata

### 4. twitter_fake_news/
**Description**: Twitter fake news dataset
**Source**: Collection of tweets labeled for fake news
**Files**:
- `tweets.csv`: Tweet data with labels

## Data Collection Guidelines

1. **Never modify raw files** - Always create processed versions
2. **Keep original format** - Preserve original structure
3. **Document sources** - Record where each dataset came from
4. **Maintain versioning** - Use git LFS for large files

## Usage Example

```python
from data.data_loader import DataLoader

loader = DataLoader()
# Load primary dataset
df_primary = loader.load_dataset("primary")
# Load Kaggle dataset
df_kaggle = loader.load_dataset("kaggle")
