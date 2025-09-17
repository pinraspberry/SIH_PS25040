import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import getpass
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 5432,
    'database': 'argo_sih',
    'username': 'postgres',  # Using postgres user as confirmed in testing
    'password': os.getenv('POSTGRES_PASSWORD', ''),  # Set via environment variable
}

def get_password():
    """Get password from environment or prompt user"""
    password = DATABASE_CONFIG['password']
    if not password:
        password = getpass.getpass("Enter PostgreSQL password for user 'postgres': ")
    return password

def get_database_url():
    """Get PostgreSQL database URL for SQLAlchemy"""
    password = os.getenv('DATABASE_PASSWORD') or os.getenv('POSTGRES_PASSWORD')
    return f"postgresql+psycopg2://{DATABASE_CONFIG['username']}:{password}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"

def get_psycopg2_url():
    """Get PostgreSQL database URL for direct psycopg2 connection"""
    password = os.getenv('DATABASE_PASSWORD') or os.getenv('POSTGRES_PASSWORD')
    return f"postgresql://{DATABASE_CONFIG['username']}:{password}@{DATABASE_CONFIG['host']}:{DATABASE_CONFIG['port']}/{DATABASE_CONFIG['database']}"

def get_db_connection():
    """Get SQLAlchemy connection"""
    try:
        engine = get_sqlalchemy_engine()
        if engine:
            return engine.connect()
        return None
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None

def get_sqlalchemy_engine():
    """Get SQLAlchemy engine"""
    try:
        engine = create_engine(get_database_url())
        return engine
    except Exception as e:
        print(f"Error creating SQLAlchemy engine: {e}")
        return None

def test_connection():
    """Test database connection"""
    try:
        engine = get_sqlalchemy_engine()
        if engine:
            with engine.connect() as conn:
                result = conn.execute(text("SELECT version();"))
                version = result.fetchone()[0]
                print(f"Connected to: {version}")
                
                # Check tables
                result = conn.execute(text("""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = 'public'
                    ORDER BY table_name;
                """))
                tables = [row[0] for row in result.fetchall()]
                print(f"Tables found: {tables}")
                
                # Check record count
                result = conn.execute(text("SELECT COUNT(*) FROM argo_profiles;"))
                count = result.fetchone()[0]
                print(f"Records in argo_profiles: {count}")
                
                return True
    except Exception as e:
        print(f"Error testing connection: {e}")
        return False

if __name__ == "__main__":
    # Test the connection when run directly
    print("Testing database connection...")
    if test_connection():
        print("✅ Database connection successful!")
    else:
        print("❌ Database connection failed!")