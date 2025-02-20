import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('app/database/microplastics_reference.db')
cursor = conn.cursor()

# Query to get all unique IDs
cursor.execute("SELECT DISTINCT ID FROM microplastics")
rows = cursor.fetchall()

# Write IDs to a .txt file
with open('output_ids.txt', 'w') as file:
    for row in rows:
        file.write(f"{row[0]}\n")  # Assuming the ID is in the first column

# Close the connection
conn.close()

print("All IDs have been written to output_ids.txt")
