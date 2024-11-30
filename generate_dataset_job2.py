import pandas as pd
import numpy as np

def generate_dataset(num_records=100):
    occupations = [
        "Software Developer", "Teacher", "Nurse", "Accountant", "Marketing Manager",
        "Chef", "Electrician", "Graphic Designer", "Lawyer", "Dentist",
        "Architect", "Police Officer", "Journalist", "Pharmacist", "Psychologist",
        "Mechanical Engineer", "Financial Analyst", "Veterinarian", "Pilot", "Librarian",
        "Photographer", "Plumber", "Social Worker", "Fitness Trainer", "Real Estate Agent",
        "Firefighter", "Translator", "Optometrist", "Carpenter", "Dietitian",
        "Web Developer", "Paramedic", "Interior Designer", "Geologist", "Zoologist",
        "Physicist", "Chemist", "Economist", "Astronomer", "Archaeologist",
        "Meteorologist", "Statistician", "Biologist", "Anthropologist", "Sociologist",
        "Historian", "Philosopher", "Linguist", "Mathematician", "Actuary",
        "Surveyor", "Urban Planner", "Landscape Architect", "Horticulturist", "Agronomist",
        "Geneticist", "Microbiologist", "Biochemist", "Neurologist", "Cardiologist",
        "Radiologist", "Anesthesiologist", "Pediatrician", "Psychiatrist", "Dermatologist",
        "Oncologist", "Surgeon", "Orthodontist", "Physiotherapist", "Occupational Therapist",
        "Speech Therapist", "Audiologist", "Chiropractor", "Podiatrist", "Acupuncturist",
        "Massage Therapist", "Nutritionist", "Personal Trainer", "Yoga Instructor", "Dance Instructor",
        "Music Teacher", "Art Teacher", "Drama Teacher", "ESL Teacher", "Special Education Teacher",
        "School Counselor", "College Professor", "Librarian", "Museum Curator", "Archivist",
        "Editor", "Technical Writer", "Copywriter", "Screenwriter", "Novelist",
        "Poet", "Journalist", "Radio Host", "TV Presenter", "Film Director",
        "Software Engineer", "Data Analyst", "Cybersecurity Specialist", "Network Administrator", "Database Administrator",
        "Sales Engineer", "Sales Manager", "Sales Representative", "Sales Associate", "Sales Coordinator",
        "Marketing Manager", "Brand Manager", "Product Manager", "Project Manager", "Operations Manager",
        "Supply Chain Manager", "Inventory Manager", "Logistics Manager", "Quality Control Manager", "Customer Service Manager",
        "Recruiter", "HR Generalist", "HR Manager", "Training Coordinator", "Compensation Analyst",
        "Risk Manager", "Investment Analyst", "Financial Planner", "Tax Accountant", "Credit Analyst",
        "Financial Advisor", "Investment Manager", "Retirement Planner", "Insurance Agent", "Real Estate Agent",
        "Event Planner", "Wedding Planner", "Personal Assistant", "Executive Assistant", "Receptionist",
        "Janitor", "Security Guard", "Custodian", "Maintenance Worker", "Security Officer",
        "Security Guard", "Custodian", "Maintenance Worker", "Security Officer"
    ]
    
    data = {
        'occupation': np.random.choice(occupations, num_records),
        'age': np.random.randint(22, 65, num_records),
        'years_experience': np.random.randint(0, 40, num_records),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], num_records),
        'technical_skills': np.random.randint(1, 11, num_records),
        'communication_skills': np.random.randint(1, 11, num_records),
        'adaptability': np.random.randint(1, 11, num_records),
        'industry_automation_level': np.random.randint(1, 11, num_records),
    }
    
    df = pd.DataFrame(data)
    
    # Convert education_level to numeric
    education_map = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
    df['education_level'] = df['education_level'].map(education_map)
    
    # Add some missing values
    df.loc[np.random.choice(df.index, 50), 'technical_skills'] = np.nan
    df.loc[np.random.choice(df.index, 50), 'communication_skills'] = np.nan
    
    return df

if __name__ == "__main__":
    dataset = generate_dataset()
    dataset.to_csv("job2.csv", index=False)
    print("Dataset generated and saved as 'job2.csv'")
    print(dataset.head())
    print("\nDataset shape:", dataset.shape)
    print("\nMissing values:\n", dataset.isnull().sum())
    print("\nUnique occupations:", dataset['occupation'].nunique())

