import sqlite3

def init_db():
    conn = sqlite3.connect('database.db')
    cursor = conn.cursor()
    
    # Create Admin table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS admin (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL
    )
    ''')
    
    # Create Operator table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS operator (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        password TEXT NOT NULL,
        name TEXT NOT NULL
    )
    ''')
    
    # Create Passenger table (8 fields)
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS passenger (
        pid TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        age INTEGER,
        gender TEXT,
        address TEXT,
        contact TEXT,
        nid TEXT,
        status TEXT DEFAULT 'Clear'
    )
    ''')
    
    # Create Recognition History table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS recognition_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        passenger_id TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        image_path TEXT,
        FOREIGN KEY (passenger_id) REFERENCES passenger (pid)
    )
    ''')
    
    # Insert default admin if not exists
    cursor.execute("INSERT OR IGNORE INTO admin (username, password) VALUES (?, ?)", ('admin', 'admin123'))
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    init_db()
