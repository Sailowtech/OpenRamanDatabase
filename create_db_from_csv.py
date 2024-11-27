#Script used to import data from a csv and creating the first data into the database

import sqlite3
import pandas as pd

# Read the CSV file with appropriate separator
df = pd.read_csv("microplastics.csv", sep=',')

# Process data into structured format
data = {}
for index, row in df.iterrows():
    ID = row.iloc[1].strip()  # Using iloc for proper positional access
    intensity = float(row.iloc[3])  # Directly convert to float without strip
    wave_number = int(row.iloc[4])  
    if ID not in data:
        data[ID] = {'Intensities': [], 'WaveNumbers': []}
    data[ID]['Intensities'].append(intensity)
    data[ID]['WaveNumbers'].append(wave_number)
    print(ID)
# Connect to SQLite database
conn = sqlite3.connect('microplastics_reference.db')
cursor = conn.cursor()

# Create a table
cursor.execute('''
CREATE TABLE IF NOT EXISTS microplastics (
    ID TEXT,
    Intensity REAL,
    WaveNumber REAL,
    Comment TEXT,
    Picture BLOB
)
''')

# Insert data into the table
for ID in data:
    for intensity, wave_number in zip(data[ID]['Intensities'], data[ID]['WaveNumbers']):
        cursor.execute("INSERT INTO microplastics (ID, Intensity, WaveNumber) VALUES (?, ?, ?)", (ID, intensity, wave_number))
    print(ID)
# Commit the changes
conn.commit()

# Function to query data by ID
def query_by_id(search_id):
    cursor.execute("SELECT Intensity, WaveNumber FROM microplastics WHERE ID=?", (search_id,))
    rows = cursor.fetchall()
    intensities, wave_numbers = zip(*rows)
    return list(intensities), list(wave_numbers)

# Example usage:
ID_to_query = "nylon PLAS190"
intensities, wave_numbers = query_by_id(ID_to_query)
print("Intensities:", intensities)
print("Wave Numbers:", wave_numbers)

# Close the connection
conn.close()