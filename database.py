import sqlite3

conn = sqlite3.connect("users.db")
c = conn.cursor()
c.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL,
        email TEXT NOT NULL,
        otp TEXT,
        password TEXT NOT NULL
        
    )
""")
conn.commit()
conn.close()

def add_user(name,email,otp, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("INSERT INTO users (name,email,otp,password) VALUES (?, ?, ?,?)", (name,email,otp, password))
    conn.commit()
    conn.close()

def authenticate_user(email, password):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ? AND password = ?", (email, password))
    user = c.fetchone()
    conn.close()
    return user
def fetch_user(email):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE email = ?", (email,))
    user = c.fetchone()
    conn.close()
    return user
def update_otp(email, otp):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("UPDATE users SET otp = ? WHERE email = ?", (otp, email))
    conn.commit()
    conn.close()

def fetch_otp(email):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT otp FROM users WHERE email = ?", (email,))
    otp = c.fetchone()
    conn.close()
    return otp