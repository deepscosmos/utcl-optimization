# from sqlalchemy import create_engine, text  # Ensure text is imported here
# from sqlalchemy.orm import sessionmaker, declarative_base  # Import declarative_base from sqlalchemy.orm

# # Database connection URL
# SQLALCHEMY_DATABASE_URL = "postgresql://postgre:PasForNothing@localhost:5432/mydb"

# # Create the database engine
# engine = create_engine(SQLALCHEMY_DATABASE_URL)

# # Create a configured "Session" class
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# # Create the base class for models
# Base = declarative_base()  # Updated import to remove deprecation warning

# # Test the connection
# if __name__ == "__main__":
#     try:
#         # Create a new session
#         with SessionLocal() as session:
#             # Run a simple test query to check connection
#             result = session.execute(text("SELECT 1")).scalar()  # Use text('SELECT 1')
#             print("Database connection successful:", result == 1)
#     except Exception as e:
#         print("Database connection failed:", e)
# from sqlalchemy import create_engine, text
# from sqlalchemy.orm import sessionmaker
# # from app.db.session import SessionLocal

# # Replace 'your_username' and 'your_password' with your actual credentials
# SQLALCHEMY_DATABASE_URL = "postgresql://postgres:PasForNothing@localhost:5432/mydb"

# # Create the database engine
# engine = create_engine(SQLALCHEMY_DATABASE_URL)

# # Test the connection
# if __name__ == "__main__":
#     try:
#         # Create a new session
#         with engine.connect() as conn:
#             # Run a simple test query to check connection
#             result = conn.execute(text("SELECT 1")).scalar()
#             print("Database connection successful:", result == 1)
#     except Exception as e:
#         print("Database connection failed:", e)
# from sqlalchemy import create_engine, text


# from sqlalchemy import create_engine,text
# from sqlalchemy.orm import sessionmaker

# # Replace 'your_username' and 'your_password' with your actual credentials
# SQLALCHEMY_DATABASE_URL = "postgresql://postgres:PasForNothing@localhost:5432/mydb"

# # Create the database engine
# engine = create_engine(SQLALCHEMY_DATABASE_URL)

# # Create SessionLocal class using sessionmaker
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# # Test the connection
# if __name__ == "__main__":
#     try:
#         # Create a new session
#         with engine.connect() as conn:
#             # Run a simple test query to check connection
#             result = conn.execute(text("SELECT 1")).scalar()
#             print("Database connection successful:", result == 1)
#     except Exception as e:
#         print("Database connection failed:", e)

   
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Database configuration
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:PasForNothing@localhost:5432/postgres"

# Create the database engine
engine = create_engine(SQLALCHEMY_DATABASE_URL)

# Create a SessionLocal class using sessionmaker
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to fetch all model names from the database
def get_model_names():
    # Use a new session for every query
    with SessionLocal() as session:
        try:
            # Use raw SQL to fetch all rows
            query = """SELECT DISTINCT "ModelName" FROM "UTCL_Optimizer"."Model_Factors";"""
            df_Master = pd.read_sql(query, session.bind)  # Fetch into a Pandas DataFrame
            print("Fetched Models from DB:", df_Master["ModelName"].tolist())  # Debugging
            return df_Master["ModelName"].tolist()  # Return model names as a list
        except Exception as e:
            print("Error fetching model names:", e)
            return []

# Function to fetch all data from the `Model_Factors` table for verification
def fetch_all_model_factors():
    with SessionLocal() as session:
        try:
            # Fetch the entire table for debugging purposes
            query = """SELECT * FROM "UTCL_Optimizer"."Model_Factors";"""
            df_Master = pd.read_sql(query, session.bind)
            print("Full Model_Factors Table:\n", df_Master)  # Debugging
            return df_Master
        except Exception as e:
            print("Error fetching all model factors:", e)
            return pd.DataFrame()

# Test the connection and fetch the data
if __name__ == "__main__":
    # Test database connection
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            print("Database connection successful:", result == 1)
    except Exception as e:
        print("Database connection failed:", e)
        exit()

    # Fetch and print all model names
    print("Fetching model names...")
    model_names = get_model_names()
    print("Model Names:", model_names)

    # Fetch and print the entire Model_Factors table for verification
    print("\nFetching the full Model_Factors table...")
    model_factors_df = fetch_all_model_factors()
    print("Model_Factors Table:\n", model_factors_df)

   