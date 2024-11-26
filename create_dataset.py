import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(50)

# Create a dataframe with 50 records
n_records = 100

data = {
    'age': np.random.randint(22, 65, n_records),
    'years_experience': np.random.randint(0, 40, n_records),
    'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_records),
    'technical_skills': np.random.randint(1, 11, n_records),  # Scale of 1-10
    'communication_skills': np.random.randint(1, 11, n_records),  # Scale of 1-10
    'adaptability': np.random.randint(1, 11, n_records),  # Scale of 1-10
    'industry_automation_level': np.random.randint(1, 11, n_records),  # Scale of 1-10
    'job_threat_level': None  # To be filled based on other features
}

df = pd.DataFrame(data)

# Convert education_level to numeric
education_map = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
df['education_level'] = df['education_level'].map(education_map)

# Add some missing values
df.loc[np.random.choice(df.index, 5), 'technical_skills'] = np.nan
df.loc[np.random.choice(df.index, 5), 'communication_skills'] = np.nan

# Calculate job threat level based on other features
df['job_threat_score'] = (
    -0.02 * df['age'] +
    -0.05 * df['years_experience'] +
    -0.5 * df['education_level'] +
    -0.2 * df['technical_skills'] +
    -0.2 * df['communication_skills'] +
    -0.2 * df['adaptability'] +
    0.5 * df['industry_automation_level']
)

# Normalize job threat score to 0-10 range
df['job_threat_score'] = (df['job_threat_score'] - df['job_threat_score'].min()) / (df['job_threat_score'].max() - df['job_threat_score'].min()) * 10

# Categorize job threat level
df['job_threat_level'] = pd.cut(df['job_threat_score'], 
                                bins=[0, 3, 7, 10], 
                                labels=['Low', 'Medium', 'High'])

# Drop the job_threat_score column as it's not needed for the final dataset
df = df.drop('job_threat_score', axis=1)

# Save the dataset to a CSV file
df.to_csv('job_threat_dataset.csv', index=False)

print("Dataset created and saved as 'job_threat_dataset.csv'")
print(df.head())
print("\nDataset shape:", df.shape)
print("\nMissing values:\n", df.isnull().sum())
print("\nJob threat level distribution:\n", df['job_threat_level'].value_counts(normalize=True))


