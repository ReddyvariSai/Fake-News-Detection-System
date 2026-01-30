import pandas as pd

# Create minimal dataset
data = {
    'title': [
        'Study Shows Benefits of Exercise',
        'Federal Reserve Maintains Rates',
        'NASA Reports Climate Data',
        'Miracle Cure Discovered',
        'Government Cover-Up Exposed'
    ],
    'text': [
        'Research published in Nature shows regular exercise reduces health risks.',
        'The Federal Reserve announced interest rates will remain unchanged.',
        'NASA satellite data indicates continued ice melt in polar regions.',
        'BREAKING: Secret herb cures all diseases overnight! Share this!',
        'SHOCKING truth about government surveillance revealed by insider!'
    ],
    'subject': ['health', 'economy', 'science', 'health', 'politics'],
    'date': ['2023-10-15', '2023-10-14', '2023-10-13', '2023-10-12', '2023-10-11'],
    'label': [0, 0, 0, 1, 1]  # 0 = real, 1 = fake
}

df = pd.DataFrame(data)
df.to_csv('fake_news.csv', index=False)
print("fake_news.csv created successfully!")
