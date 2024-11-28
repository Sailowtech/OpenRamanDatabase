#Exemple on how to add extra information to the database.
import sqlite3

# Define the path to your database
db_file_path = 'app/database/microplastics_reference.db'

def update_database_structure():
    try:
        # Connect to the SQLite database
        conn = sqlite3.connect(db_file_path)
        cursor = conn.cursor()

        # SQL command to add a new column called "Comment" to the "Data" table
        cursor.execute("ALTER TABLE sample_bank ADD COLUMN similarity_score TEXT")
        
        # Commit the changes
        conn.commit()
        print("Database structure updated successfully.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the database connection
        if conn:
            conn.close()


def add_table():
    # Connect to SQLite database
    conn = sqlite3.connect(db_file_path)
    cursor = conn.cursor()


    # Create a sample_bank table if it doesn't exist
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS sample_bank (
        sample_id TEXT,
        intensity REAL,
        wave_number REAL,
        Comment TEXT
    )
    ''')

    conn.commit()
    conn.close()
    print("Created Table")

#add_table()
#
update_database_structure()