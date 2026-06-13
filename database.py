import sqlite3
from datetime import datetime
import hashlib

def get_db_connection():
    conn = sqlite3.connect('diabetes_app.db')
    conn.row_factory = sqlite3.Row
    return conn

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def create_database():
    """Buat tabel jika belum ada (TIDAK drop tabel yang sudah ada)."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            nama TEXT,
            email TEXT,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            result TEXT,
            probability REAL,
            glukosa REAL,
            bmi REAL,
            usia INTEGER,
            tekanan_darah REAL,
            insulin REAL,
            kulit REAL,
            kehamilan INTEGER,
            dpf REAL,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Hanya insert admin jika belum ada
    cursor.execute("SELECT id FROM users WHERE username = 'admin'")
    if not cursor.fetchone():
        cursor.execute('''
            INSERT INTO users (username, password, nama, email, role)
            VALUES (?, ?, ?, ?, ?)
        ''', ('admin', hash_password('admin123'), 'Administrator', 'admin@diabetes.ai', 'admin'))

    conn.commit()
    conn.close()

def check_login(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    hashed_pass = hash_password(password)
    cursor.execute(
        "SELECT id, username, nama, email, role FROM users WHERE username = ? AND password = ?",
        (username, hashed_pass)
    )
    user = cursor.fetchone()
    if user:
        cursor.execute("UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?", (user[0],))
        conn.commit()
        result = (user[0], user[1], user[2], user[3], user[4])
        conn.close()
        return result
    conn.close()
    return None

def add_user(username, password, nama, email, role='user'):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute('''
            INSERT INTO users (username, password, nama, email, role)
            VALUES (?, ?, ?, ?, ?)
        ''', (username, hash_password(password), nama, email, role))
        conn.commit()
        conn.close()
        return True
    except sqlite3.IntegrityError:
        conn.close()
        return False

def get_all_users():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, username, nama, email, role, created_at, last_login FROM users ORDER BY id')
    users = cursor.fetchall()
    conn.close()
    return [tuple(u) for u in users]

def delete_user(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM users WHERE id = ?", (user_id,))
    cursor.execute("DELETE FROM predictions WHERE user_id = ?", (user_id,))
    conn.commit()
    conn.close()

def save_prediction(user_id, username, data, result, probability):
    conn = get_db_connection()
    cursor = conn.cursor()
    # probability adalah array [prob_negatif, prob_positif]
    prob_value = float(probability[1]) if result == "Positif" else float(probability[0])
    cursor.execute('''
        INSERT INTO predictions
        (user_id, username, result, probability, glukosa, bmi, usia,
         tekanan_darah, insulin, kulit, kehamilan, dpf)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        user_id, username, result, prob_value,
        data['glukosa'], data['bmi'], data['usia'],
        data['tekanan_darah'], data['insulin'], data['kulit'],
        data['kehamilan'], data['dpf']
    ))
    conn.commit()
    conn.close()

def get_user_history(user_id, limit=50):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, prediction_date, result, probability, glukosa, bmi, usia,
               tekanan_darah, insulin, kehamilan, kulit, dpf
        FROM predictions WHERE user_id = ?
        ORDER BY prediction_date DESC LIMIT ?
    ''', (user_id, limit))
    rows = cursor.fetchall()
    conn.close()
    return [tuple(r) for r in rows]

def get_all_predictions():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        SELECT id, username, prediction_date, result, probability, glukosa, bmi, usia
        FROM predictions ORDER BY prediction_date DESC
    ''')
    rows = cursor.fetchall()
    conn.close()
    return [tuple(r) for r in rows]

def get_statistics():
    """Bug fix: fetchone() dipanggil sekali per query."""
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM predictions")
    total = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE result = 'Positif'")
    positif = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM predictions WHERE result = 'Negatif'")
    negatif = cursor.fetchone()[0]

    conn.close()
    return total, positif, negatif

def get_monthly_stats():
    """Ambil data tren 6 bulan terakhir untuk grafik dashboard."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT strftime('%Y-%m', prediction_date) as bulan,
               COUNT(*) as total,
               SUM(CASE WHEN result='Positif' THEN 1 ELSE 0 END) as positif,
               SUM(CASE WHEN result='Negatif' THEN 1 ELSE 0 END) as negatif
        FROM predictions
        GROUP BY bulan
        ORDER BY bulan DESC
        LIMIT 6
    """)
    rows = cursor.fetchall()
    conn.close()
    return [tuple(r) for r in reversed(rows)]
