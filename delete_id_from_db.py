import sqlite3

# Connect to SQLite database
conn = sqlite3.connect('app/database/microplastics_reference.db')
cursor = conn.cursor()

# Delete data for the specific ID
id_to_delete = "algin BIOL006"
cursor.execute("DELETE FROM microplastics WHERE ID=?", (id_to_delete,))

# Commit the changes
conn.commit()

# Check how many rows were deleted (optional)
deleted_rows = cursor.rowcount
print(f"Deleted {deleted_rows} rows with ID '{id_to_delete}'.")

# Close the connection
conn.close()
