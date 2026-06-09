import sqlite3
import hashlib
from datetime import datetime

DB_NAME = "diabetes_app.db"

def create_database():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            nama_lengkap TEXT,
            email TEXT,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS prediksi_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            usia INTEGER,
            bmi REAL,
            kehamilan INTEGER,
            kulit INTEGER,
            glukosa INTEGER,
            tekanan_darah INTEGER,
            insulin INTEGER,
            dpf REAL,
            hasil TEXT,
            probabilitas_rendah REAL,
            probabilitas_tinggi REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database berhasil dibuat")

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def add_user(username, password, nama_lengkap="", email="", role="user"):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    try:
        hashed_pw = hash_password(password)
        cursor.execute('''
            INSERT INTO users (username, password, nama_lengkap, email, role)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, hashed_pw, nama_lengkap, email, role))
        conn.commit()
        print(f"User {username} berhasil ditambahkan")
        return True
    except sqlite3.IntegrityError:
        print(f"Username {username} sudah ada")
        return False
    finally:
        conn.close()

def check_login(username, password):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    hashed_pw = hash_password(password)
    cursor.execute('''
        SELECT id, username, nama_lengkap, email, role FROM users 
        WHERE username = ? AND password = ?
    ''', (username, hashed_pw))
    
    user = cursor.fetchone()
    
    if user:
        cursor.execute('''
            UPDATE users SET last_login = ? WHERE id = ?
        ''', (datetime.now(), user[0]))
        conn.commit()
    
    conn.close()
    return user

def get_user_by_username(username):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, nama_lengkap, email, role, created_at, last_login 
        FROM users WHERE username = ?
    ''', (username,))
    
    user = cursor.fetchone()
    conn.close()
    return user

def save_prediction(user_id, username, data, hasil, proba):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO prediksi_history (
            user_id, username, usia, bmi, kehamilan, kulit, 
            glukosa, tekanan_darah, insulin, dpf, 
            hasil, probabilitas_rendah, probabilitas_tinggi
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, username,
        data['usia'], data['bmi'], data['kehamilan'], data['kulit'],
        data['glukosa'], data['tekanan_darah'], data['insulin'], data['dpf'],
        hasil, proba[0], proba[1]
    ))
    
    conn.commit()
    conn.close()

def get_user_history(user_id, limit=10):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, created_at, hasil, probabilitas_tinggi, 
               glukosa, bmi, usia
        FROM prediksi_history 
        WHERE user_id = ? 
        ORDER BY created_at DESC 
        LIMIT ?
    ''', (user_id, limit))
    
    history = cursor.fetchall()
    conn.close()
    return history

def get_all_users():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT id, username, nama_lengkap, email, role, created_at, last_login 
        FROM users ORDER BY id
    ''')
    
    users = cursor.fetchall()
    conn.close()
    return users

def delete_user(user_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('DELETE FROM prediksi_history WHERE user_id = ?', (user_id,))
    cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
    
    conn.commit()
    conn.close()

def update_user_role(user_id, role):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute('UPDATE users SET role = ? WHERE id = ?', (role, user_id))
    conn.commit()
    conn.close()

if __name__ == "__main__":
    create_database()
    add_user("admin", "admin123", "Administrator", "admin@diabetes.com", "admin")
    add_user("dokter", "dokter123", "Dokter Spesialis", "dokter@diabetes.com", "dokter")
    add_user("pasien", "pasien123", "Pasien Umum", "pasien@diabetes.com", "user")
    add_user("user", "user123", "Pengguna Biasa", "user@diabetes.com", "user")
    print("User default berhasil ditambahkan")