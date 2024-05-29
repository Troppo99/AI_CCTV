import mysql.connector

# Establish a connection to the database
conn = mysql.connector.connect(
    host="localhost",  # Your MySQL host
    user="root",  # Your MySQL username
    password="robot123",  # Your MySQL password
    database="mydb",  # The name of the database
)

# Create a cursor object
cursor = conn.cursor()

# Define the data to be inserted
name = "John Doe"
email = "john.doe@example.com"

# Create the SQL query
query = "INSERT INTO users (name, email) VALUES (%s, %s)"

# Execute the query
cursor.execute(query, (name, email))

# Commit the transaction
conn.commit()

# Print the ID of the inserted row
print(f"Inserted row ID: {cursor.lastrowid}")

# Close the cursor and connection
cursor.close()
conn.close()
