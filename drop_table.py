#Exemple on how to drop a table from the database
import sqlite3

# Path to your SQLite database
db_file_path = 'app/database/microplastics_reference.db'  # Make sure this path is correct

# Connect to the SQLite database
conn = sqlite3.connect(db_file_path)
cursor = conn.cursor()

# Drop the 'slopp' table
try:
    cursor.execute('DROP TABLE IF EXISTS sloppe')
    print("Table 'sloppe' has been dropped successfully.")
except sqlite3.Error as e:
    print(f"An error occurred while dropping the table: {e}")

# Commit changes and close the connection
conn.commit()
conn.close()