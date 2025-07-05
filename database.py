import sqlite3
import logging
from flask import redirect, url_for, flash # Keep flask imports if used (not directly here now)
import os
from datetime import date

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Specify the database path
DATABASE_PATH = 'blood_donation_enhanced.db' # New name to avoid conflicts

def get_db_connection(database_path=DATABASE_PATH):
    """Create a database connection with row factory"""
    try:
        conn = sqlite3.connect(database_path)
        conn.row_factory = sqlite3.Row # Return rows as dictionary-like objects
        conn.execute("PRAGMA foreign_keys = ON") # Enforce foreign keys
        return conn
    except sqlite3.Error as e:
        logger.error(f"Error connecting to database '{database_path}': {e}")
        raise # Re-raise the exception

def init_db(database_path=DATABASE_PATH):
    """Initialize database tables if they don't exist"""
    if is_database_initialized(database_path):
        logger.info("Database already initialized.")
        return

    logger.info(f"Initializing database at '{database_path}'...")
    conn = None
    try:
        conn = get_db_connection(database_path)
        # Updated donors table with new fields
        conn.execute('''
        CREATE TABLE IF NOT EXISTS donors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            gender TEXT NOT NULL CHECK(gender IN ('M', 'F', 'O')), -- Added constraint
            blood_type TEXT NOT NULL,
            weight REAL,
            hemoglobin_level REAL,
            blood_pressure TEXT, -- Store as text e.g., '120/80'
            last_donated DATE, -- Store as YYYY-MM-DD text
            alcoholic_status TEXT CHECK(alcoholic_status IN ('Never', 'Occasional', 'Regular', 'Recovering', 'Unknown')), -- Added constraint
            travelled_recently INTEGER DEFAULT 0, -- Boolean 0/1
            travel_details TEXT,
            had_recent_illness INTEGER DEFAULT 0, -- Boolean 0/1
            illness_details TEXT,
            recent_tattoo_piercing INTEGER DEFAULT 0, -- Boolean 0/1
            is_pregnant_breastfeeding INTEGER DEFAULT 0, -- Boolean 0/1 (relevant for F)
            medical_conditions TEXT, -- General existing conditions
            medications TEXT,
            registration_date DATE DEFAULT CURRENT_DATE -- Store as YYYY-MM-DD text
        )
        ''')

        # Create donations table (schema remains the same for now)
        conn.execute('''
        CREATE TABLE IF NOT EXISTS donations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            donor_id INTEGER NOT NULL,
            donation_date DATE NOT NULL,
            donation_volume REAL,
            hemoglobin_pre REAL,
            hemoglobin_post REAL,
            blood_pressure_pre TEXT,
            blood_pressure_post TEXT,
            eligibility_status TEXT,
            notes TEXT,
            FOREIGN KEY (donor_id) REFERENCES donors(id) ON DELETE CASCADE
        )
        ''')

        conn.commit()
        logger.info("Database initialized successfully with new schema")
    except sqlite3.Error as e:
        logger.error(f"Error initializing database: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()

def add_donor(donor_data, database_path=DATABASE_PATH):
    """Add a new donor record with updated fields"""
    conn = None
    sql = """
        INSERT INTO donors (
            name, age, gender, blood_type, weight, hemoglobin_level, blood_pressure,
            last_donated, alcoholic_status, travelled_recently, travel_details,
            had_recent_illness, illness_details, recent_tattoo_piercing,
            is_pregnant_breastfeeding, medical_conditions, medications
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = (
        donor_data['name'], donor_data['age'], donor_data['gender'], donor_data['blood_type'],
        donor_data.get('weight'), donor_data.get('hemoglobin_level'), donor_data.get('blood_pressure'),
        donor_data.get('last_donated'), donor_data.get('alcoholic_status', 'Unknown'), # Default if missing
        donor_data.get('travelled_recently', 0), donor_data.get('travel_details'),
        donor_data.get('had_recent_illness', 0), donor_data.get('illness_details'),
        donor_data.get('recent_tattoo_piercing', 0),
        donor_data.get('is_pregnant_breastfeeding', 0),
        donor_data.get('medical_conditions'), donor_data.get('medications')
    )

    try:
        conn = get_db_connection(database_path)
        cursor = conn.cursor()
        cursor.execute(sql, params)
        conn.commit()
        donor_id = cursor.lastrowid
        logger.info(f"Added new donor with ID: {donor_id}")
        return donor_id
    except sqlite3.Error as e:
        logger.error(f"Error adding donor: {e}")
        if conn:
            conn.rollback()
        return None # Indicate failure
    finally:
        if conn:
            conn.close()

def get_all_donors(database_path=DATABASE_PATH):
    """Retrieve all donors sorted by registration date"""
    conn = None
    try:
        conn = get_db_connection(database_path)
        # Selecting all columns now
        donors = conn.execute(
            "SELECT * FROM donors ORDER BY registration_date DESC, name ASC"
        ).fetchall()
        return donors
    except sqlite3.Error as e:
        logger.error(f"Error fetching all donors: {e}")
        return [] # Return empty list on error
    finally:
        if conn:
            conn.close()

def get_donor_by_id(donor_id, database_path=DATABASE_PATH):
    """Retrieve a single donor by their ID"""
    conn = None
    try:
        conn = get_db_connection(database_path)
        # Selecting all columns now
        donor = conn.execute(
            "SELECT * FROM donors WHERE id = ?", (donor_id,)
        ).fetchone()
        return donor
    except sqlite3.Error as e:
        logger.error(f"Error fetching donor with ID {donor_id}: {e}")
        return None # Indicate failure or not found
    finally:
        if conn:
            conn.close()

def is_database_initialized(database_path=DATABASE_PATH):
    """Check if database tables (donors) exist"""
    if not os.path.exists(database_path):
        return False
    conn = None
    try:
        conn = get_db_connection(database_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='donors'")
        donors_table = cursor.fetchone()
        return donors_table is not None
    except sqlite3.Error as e:
        logger.error(f"Error checking database initialization: {e}")
        return False
    finally:
        if conn:
            conn.close()

# Ensure initialization happens if module is run directly (optional)
if __name__ == '__main__':
    print(f"Checking database '{DATABASE_PATH}'...")
    init_db()
    print("Database check/initialization complete.")
    # Example usage (optional)
    # donors = get_all_donors()
    # print(f"Found {len(donors)} donors.")