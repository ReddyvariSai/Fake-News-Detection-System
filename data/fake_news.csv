import pandas as pd
import numpy as np

# Create sample data directly
sample_data = [
    # Real News Examples
    {
        'title': 'New Study Shows Benefits of Mediterranean Diet',
        'text': 'A comprehensive study published in the New England Journal of Medicine followed 10,000 participants over 5 years and found that adherence to a Mediterranean diet was associated with a 30% reduction in cardiovascular events. The research was conducted by Harvard University and peer-reviewed by independent experts.',
        'subject': 'health',
        'date': '2023-10-15',
        'label': 0
    },
    {
        'title': 'Federal Reserve Announces Interest Rate Decision',
        'text': 'The Federal Reserve announced today that it will maintain interest rates at current levels, citing stable inflation and employment data. Chair Jerome Powell stated that the economy shows continued growth with moderate inflation expectations for the coming quarter.',
        'subject': 'economy',
        'date': '2023-10-14',
        'label': 0
    },
    {
        'title': 'NASA Releases New Climate Change Data',
        'text': 'NASA scientists have released new satellite data showing continued ice melt in Antarctica. The data, collected over 10 years, indicates an acceleration in glacial retreat. The findings were published in the journal Science after rigorous peer review.',
        'subject': 'science',
        'date': '2023-10-13',
        'label': 0
    },
    {
        'title': 'WHO Updates COVID-19 Vaccination Guidelines',
        'text': 'The World Health Organization has updated its COVID-19 vaccination recommendations based on new clinical trial data. The guidelines now recommend booster shots for high-risk populations. The decision was made after analyzing data from multiple countries.',
        'subject': 'health',
        'date': '2023-10-12',
        'label': 0
    },
    {
        'title': 'Renewable Energy Costs Continue to Decline',
        'text': 'According to a report from the International Renewable Energy Agency, solar and wind power costs have decreased by 40% over the past five years. The data shows that renewable energy is now cost-competitive with fossil fuels in most markets worldwide.',
        'subject': 'environment',
        'date': '2023-10-11',
        'label': 0
    },
    
    # Fake News Examples
    {
        'title': 'BREAKING: Miracle Herb Cures Cancer Overnight!',
        'text': 'SHOCKING discovery: A secret herb from the Amazon rainforest has been found to CURE all types of cancer in just 24 hours! Big Pharma is trying to SUPPRESS this information to protect their profits. Doctors are FURIOUS about this natural remedy! Share this before it gets deleted!',
        'subject': 'health',
        'date': '2023-10-15',
        'label': 1
    },
    {
        'title': 'You Won\'t Believe What Celebrity Said About Vaccines!',
        'text': 'SECRET recording reveals shocking truth about vaccines that the mainstream media doesn\'t want you to know! A famous Hollywood star claims vaccines contain microchips for population control. This information is being SUPPRESSED by the government!',
        'subject': 'entertainment',
        'date': '2023-10-14',
        'label': 1
    },
    {
        'title': 'Government Cover-Up Exposed by Whistleblower',
        'text': 'ANONYMOUS insider reveals SECRET government program to control population through 5G technology! The truth about mind control waves is finally exposed. They have been hiding this technology for years! Act now before it\'s too late!',
        'subject': 'technology',
        'date': '2023-10-13',
        'label': 1
    },
    {
        'title': 'Doctors Hate This Simple Weight Loss Trick',
        'text': 'LOSE 20 pounds in ONE WEEK with this simple trick that doctors don\'t want you to know! The medical establishment is SUPPRESSING this information to keep you buying expensive drugs. Click here to learn the secret that Big Pharma hates!',
        'subject': 'health',
        'date': '2023-10-12',
        'label': 1
    },
    {
        'title': 'Aliens Contact Earth Leaders in Secret Meeting',
        'text': 'BREAKING: Extraterrestrial beings have made contact with world leaders in a SECRET meeting at Area 51! The government is HIDING the truth about alien technology that could solve our energy crisis. This information is CLASSIFIED! Share with everyone!',
        'subject': 'science',
        'date': '2023-10-11',
        'label': 1
    }
]

# Create more samples by modifying existing ones
def create_larger_dataset(base_samples, target_size=100):
    """Expand the dataset to target size"""
    expanded_data = base_samples.copy()
    
    subjects = ['politics', 'health', 'science', 'technology', 'business', 
                'entertainment', 'sports', 'education', 'environment', 'world']
    
    # Templates for generating more data
    real_templates = [
        "New research from {university} shows that {finding}. The study was published in {journal}.",
        "Official data indicates that {metric} has {change}. Experts attribute this to {reason}.",
        "According to {expert}, a leading authority in {field}, {statement}.",
        "The {organization} announced today that {announcement}. This decision follows {process}.",
        "A recent survey found that {percentage}% of {group} reported {result}. The margin of error is {error}%."
    ]
    
    fake_templates = [
        "SHOCKING: {secret} REVEALED! {entity} has been HIDING this for years!",
        "You won't BELIEVE what {source} says about {topic}! The TRUTH will shock you!",
        "{product} INSTANTLY cures {problem}! {industry} is FURIOUS about this discovery!",
        "BREAKING: {conspiracy} COVER-UP exposed by {whistleblower}! Share before deletion!",
        "Doctors HATE this simple trick for {benefit}! {industry} tries to SUPPRESS it!"
    ]
    
    universities = ['Harvard', 'Stanford', 'MIT', 'Cambridge', 'Oxford', 'Tokyo University']
    journals = ['Nature', 'Science', 'The Lancet', 'JAMA', 'Cell', 'PNAS']
    experts = ['Dr. Sarah Johnson', 'Prof. Michael Chen', 'Dr. Lisa Wang', 'Prof. Robert Kim']
    organizations = ['WHO', 'CDC', 'FDA', 'UN', 'World Bank', 'IMF']
    
    # Generate more real news
    while len(expanded_data) < target_size // 2:
        template = np.random.choice(real_templates)
        subject = np.random.choice(subjects)
        
        if '{university}' in template:
            text = template.format(
                university=np.random.choice(universities),
                finding='regular exercise reduces heart disease risk by 25%',
                journal=np.random.choice(journals)
            )
        elif '{metric}' in template:
            metrics = ['unemployment', 'inflation', 'GDP growth', 'carbon emissions']
            text = template.format(
                metric=np.random.choice(metrics),
                change='decreased significantly',
                reason='policy interventions'
            )
        else:
            text = template.format(
                expert=np.random.choice(experts),
                field='climate science',
                statement='immediate action is needed to address environmental concerns'
            )
        
        expanded_data.append({
            'title': f'New {subject.capitalize()} Research Findings',
            'text': text + ' The methodology was rigorous and results were statistically significant.',
            'subject': subject,
            'date': f'2023-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}',
            'label': 0
        })
    
    # Generate more fake news
    while len(expanded_data) < target_size:
        template = np.random.choice(fake_templates)
        subject = np.random.choice(subjects)
        
        if '{secret}' in template:
            text = template.format(
                secret='government surveillance program',
                entity='the authorities'
            )
        elif '{product}' in template:
            text = template.format(
                product='this one simple vitamin',
                problem='all chronic diseases',
                industry='Big Pharma'
            )
        else:
            text = template.format(
                source='anonymous insider',
                topic='the real cause of climate change'
            )
        
        expanded_data.append({
            'title': f'SHOCKING {subject.upper()} REVELATION!',
            'text': text + ' SHARE THIS WITH EVERYONE BEFORE THEY DELETE IT! The mainstream media is covering this up!',
            'subject': subject,
            'date': f'2023-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}',
            'label': 1
        })
    
    return expanded_data

# Create the full dataset
full_dataset = create_larger_dataset(sample_data, 100)

# Convert to DataFrame
df = pd.DataFrame(full_dataset)

# Save to CSV
df.to_csv('fake_news.csv', index=False, encoding='utf-8')

print(f"Dataset created with {len(df)} samples")
print(f"Real news: {len(df[df['label'] == 0])} samples")
print(f"Fake news: {len(df[df['label'] == 1])} samples")
print("\nFirst 10 samples:")
print(df[['title', 'subject', 'label']].head(10).to_string())
