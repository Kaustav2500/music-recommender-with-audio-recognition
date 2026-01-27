import os
import sys

# add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_connect import get_db_connection

try:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM songs")
    count = cursor.fetchone()[0]


    print(f"Database contains {count} songs.")

    if count > 0:
        cursor.execute("SELECT song_name, artist FROM songs LIMIT 3")
        print("Sample data:")
        for row in cursor.fetchall():
            print(f"- {row[0]} by {row[1]}")

    cursor.close()
    conn.close()

except Exception as e:
    print(f"Connection Failed: {e}")