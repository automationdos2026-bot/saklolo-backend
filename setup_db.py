import sqlite3

def init_station_db():
    conn = sqlite3.connect("stations.db")
    cursor = conn.cursor()

    # Create Stations Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS response_stations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            type TEXT NOT NULL,  -- POLICE, FIRE, AMBULANCE
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            contact TEXT NOT NULL
        )
    """)

    # Clear existing data for fresh insertion
    cursor.execute("DELETE FROM response_stations")

    # Insert Real/Sample Station Data (Legazpi City & Regional Coverage)
    sample_stations = [
        ("Legazpi City Police Station", "POLICE", 13.1398, 123.7431, "0917-111-2222"),
        ("PNP Regional Office 5 (Camp Simeon Ola)", "POLICE", 13.1465, 123.7485, "0917-333-4444"),
        ("BFP Fire Station Legazpi", "FIRE", 13.1362, 123.7450, "0918-555-6666"),
        ("BFP Daraga Station", "FIRE", 13.1520, 123.7150, "0918-777-8888"),
        ("Bicol Regional Hospital & Medical Center", "AMBULANCE", 13.1425, 123.7380, "0920-999-0000"),
        ("Aguinaldo Medical Hospital", "AMBULANCE", 13.1350, 123.7320, "0920-123-4567")
    ]

    cursor.executemany("""
        INSERT INTO response_stations (name, type, latitude, longitude, contact)
        VALUES (?, ?, ?, ?, ?)
    """, sample_stations)

    conn.commit()
    conn.close()
    print("Database 'stations.db' successfully created and populated!")

if __name__ == "__main__":
    init_station_db()