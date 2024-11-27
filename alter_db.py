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
        cursor.execute("ALTER TABLE microplastics ADD COLUMN Comment TEXT")
        
        # Commit the changes
        conn.commit()
        print("Database structure updated successfully.")

    except sqlite3.Error as e:
        print(f"An error occurred: {e}")

    finally:
        # Close the database connection
        if conn:
            conn.close()

# Call the function to update the database structure
update_database_structure()