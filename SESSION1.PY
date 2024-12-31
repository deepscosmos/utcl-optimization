# import pandas as pd
# from sqlalchemy import create_engine, text
# from sqlalchemy.orm import sessionmaker

# # Database configuration
# SQLALCHEMY_DATABASE_URL = "postgresql://postgres:PasForNothing@localhost:5432/mydb"

# # Create the database engine
# engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)

# # Create a SessionLocal class using sessionmaker
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# # Function to fetch all model names from the database
# def get_model_names():
#     try:
#         # Create a new session for fresh data
#         with SessionLocal() as session:
#             # Force refresh by querying again and closing the session immediately
#             query = """SELECT DISTINCT "ModelName" FROM "UTCL_Optimizer"."Model_Factors";"""
#             df_Master = pd.read_sql(query, session.bind)  # Fetch into a Pandas DataFrame
#             print("Fetched Models from DB:", df_Master["ModelName"].tolist())  # Debugging
#             return df_Master["ModelName"].tolist()  # Return model names as a list
#     except Exception as e:
#         print("Error fetching model names:", e)
#         return []

# # Function to commit and refresh the session
# def refresh_and_commit_session(session):
#     try:
#         # Explicitly refresh and commit the session to ensure all updates are saved
#         session.commit()
#         session.refresh()
#     except Exception as e:
#         print("Error refreshing session:", e)

# # Function to fetch all data from the `Model_Factors` table for verification
# def fetch_all_model_factors():
#     try:
#         # Create a new session for fresh data
#         with SessionLocal() as session:
#             # Force the session to refresh the data by explicitly executing raw SQL
#             query = """SELECT * FROM "UTCL_Optimizer"."Model_Factors";"""
#             df_Master = pd.read_sql(query, session.bind)  # Fetch into a Pandas DataFrame
#             print("Full Model_Factors Table:\n", df_Master)  # Debugging
#             return df_Master
#     except Exception as e:
#         print("Error fetching all model factors:", e)
#         return pd.DataFrame()

# # Test the connection and fetch the data
# if __name__ == "__main__":
#     # Test database connection
#     try:
#         with engine.connect() as conn:
#             result = conn.execute(text("SELECT 1")).scalar()
#             print("Database connection successful:", result == 1)
#     except Exception as e:
#         print("Database connection failed:", e)
#         exit()

#     # Fetch and print all model names
#     print("Fetching model names...")
#     model_names = get_model_names()
#     print("Model Names:", model_names)

#     # Fetch and print the entire Model_Factors table for verification
#     print("\nFetching the full Model_Factors table...")
#     model_factors_df = fetch_all_model_factors()
#     print("Model_Factors Table:\n", model_factors_df)

# import sqlite3

# # Close any existing connection
# connection.close()

# # Reconnect to the database (ensure this path is correct)
# connection = sqlite3.connect('postgresql://postgres:PasForNothing@localhost:5432/mydb')
# cursor = connection.cursor()

# # Fetch data again
# cursor.execute("SELECT * FROM Model_Factors;")
# rows = cursor.fetchall()
# print(f"Fetched Rows: {rows}")


# import pandas as pd
# from sqlalchemy import create_engine, text
# from sqlalchemy.orm import sessionmaker

# # Database configuration
# SQLALCHEMY_DATABASE_URL = "postgresql://postgres:PasForNothing@localhost:5432/mydb"

# # Create the database engine
# engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)

# # Create a SessionLocal class using sessionmaker
# SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# # Function to set the isolation level
# def set_isolation_level(session, level="READ COMMITTED"):
#     try:
#         # Set the isolation level to the desired level, wrapped in text()
#         session.execute(text(f"SET TRANSACTION ISOLATION LEVEL {level};"))
#         print(f"Transaction Isolation Level set to {level}")
#     except Exception as e:
#         print(f"Error setting isolation level: {e}")

# # Function to fetch all model names from the database
# def get_model_names():
#     try:
#         # Create a new session for fresh data
#         with SessionLocal() as session:
#             # Set the isolation level to ensure fresh data visibility
#             set_isolation_level(session)

#             # Force refresh by querying again and closing the session immediately
#             query = """SELECT DISTINCT "ModelName" FROM "UTCL_Optimizer"."Model_Factors";"""
#             df_Master = pd.read_sql(query, session.bind)  # Fetch into a Pandas DataFrame
#             print("Fetched Models from DB:", df_Master["ModelName"].tolist())  # Debugging
#             return df_Master["ModelName"].tolist()  # Return model names as a list
#     except Exception as e:
#         print("Error fetching model names:", e)
#         return []

# # Function to commit and refresh the session
# def refresh_and_commit_session(session):
#     try:
#         # Explicitly refresh and commit the session to ensure all updates are saved
#         session.commit()
#         session.refresh()
#     except Exception as e:
#         print("Error refreshing session:", e)

# # Function to fetch all data from the `Model_Factors` table for verification
# def fetch_all_model_factors():
#     try:
#         # Create a new session for fresh data
#         with SessionLocal() as session:
#             # Set the isolation level to ensure fresh data visibility
#             set_isolation_level(session)

#             # Force the session to refresh the data by explicitly executing raw SQL
#             query = """SELECT * FROM "UTCL_Optimizer"."Model_Factors";"""
#             df_Master = pd.read_sql(query, session.bind)  # Fetch into a Pandas DataFrame
#             print("Full Model_Factors Table:\n", df_Master)  # Debugging
#             return df_Master
#     except Exception as e:
#         print("Error fetching all model factors:", e)
#         return pd.DataFrame()

# # Test the connection and fetch the data
# if __name__ == "__main__":
#     # Test database connection
#     try:
#         with engine.connect() as conn:
#             result = conn.execute(text("SELECT 1")).scalar()
#             print("Database connection successful:", result == 1)
#     except Exception as e:
#         print("Database connection failed:", e)
#         exit()

#     # Fetch and print all model names
#     print("Fetching model names...")
#     model_names = get_model_names()
#     print("Model Names:", model_names)

#     # Fetch and print the entire Model_Factors table for verification
#     print("\nFetching the full Model_Factors table...")
#     model_factors_df = fetch_all_model_factors()
#     print("Model_Factors Table:\n", model_factors_df)

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# PostgreSQL connection configuration
SQLALCHEMY_DATABASE_URL = "postgresql://postgres:PasForNothing@localhost:5432/postgres"

# Create the database engine for PostgreSQL
engine = create_engine(SQLALCHEMY_DATABASE_URL, pool_pre_ping=True)

# Create a SessionLocal class using sessionmaker for PostgreSQL
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Function to set the isolation level for PostgreSQL
def set_isolation_level(session, level="READ COMMITTED"):
    try:
        # Set the isolation level to the desired level
        session.execute(text(f"SET TRANSACTION ISOLATION LEVEL {level};"))
        print(f"Transaction Isolation Level set to {level}")
    except Exception as e:
        print(f"Error setting isolation level: {e}")

# Function to fetch all model names from PostgreSQL database
def get_model_names():
    try:
        # Create a new session for fresh data from PostgreSQL
        with SessionLocal() as session:
            # Set the isolation level to ensure fresh data visibility
            set_isolation_level(session)

            # Force refresh by querying again
            query = """SELECT DISTINCT ModelName FROM "UTCL_Optimizer"."Model_Factors";"""
            df_Master = pd.read_sql(query, session.bind)  # Fetch into a Pandas DataFrame
            print("Fetched Models from PostgreSQL DB:", df_Master["ModelName"].tolist())  # Debugging
            return df_Master["ModelName"].tolist()  # Return model names as a list
    except Exception as e:
        print("Error fetching model names from PostgreSQL:", e)
        return []

# Function to fetch all data from the `Model_Factors` table for PostgreSQL verification
def fetch_all_model_factors():
    try:
        # Create a new session for fresh data from PostgreSQL
        with SessionLocal() as session:
            # Set the isolation level to ensure fresh data visibility
            set_isolation_level(session)

            # Force the session to refresh the data by explicitly executing raw SQL
            query = """SELECT * FROM "UTCL_Optimizer"."Model_Factors";"""
            df_Master = pd.read_sql(query, session.bind)  # Fetch into a Pandas DataFrame
            print("Full Model_Factors Table from PostgreSQL:\n", df_Master)  # Debugging
            return df_Master
    except Exception as e:
        print("Error fetching all model factors from PostgreSQL:", e)
        return pd.DataFrame()

# Function to close the PostgreSQL connection
def close_connection():
    try:
        engine.dispose()  # Close all connections in the pool
        print("PostgreSQL engine connection pool closed.")
    except Exception as e:
        print("Error closing PostgreSQL engine:", e)

# Test the PostgreSQL connection, fetch data, and close the connection
if __name__ == "__main__":
    # Test database connection for PostgreSQL
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            print("PostgreSQL Database connection successful:", result == 1)
    except Exception as e:
        print("PostgreSQL Database connection failed:", e)
        exit()

    # Fetch and print all model names from PostgreSQL
    print("Fetching model names from PostgreSQL...")
    model_names = get_model_names()
    print("Model Names from PostgreSQL:", model_names)

    # Fetch and print the entire Model_Factors table for verification from PostgreSQL
    print("\nFetching the full Model_Factors table from PostgreSQL...")
    model_factors_df = fetch_all_model_factors()
    print("Model_Factors Table from PostgreSQL:\n", model_factors_df)

    # Close the connection and cleanup
    close_connection()

    # Reopen the connection and fetch the data again
    print("\nReopening the connection and fetching data again...")

    # Test the PostgreSQL connection again after closing
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            print("PostgreSQL Database connection successful after reopen:", result == 1)
    except Exception as e:
        print("PostgreSQL Database connection failed after reopen:", e)
        exit()

    # Fetch and print all model names from PostgreSQL again
    print("Fetching model names from PostgreSQL again...")
    model_names = get_model_names()
    print("Model Names from PostgreSQL:", model_names)

    # Fetch and print the entire Model_Factors table for verification from PostgreSQL again
    print("\nFetching the full Model_Factors table from PostgreSQL again...")
    model_factors_df = fetch_all_model_factors()
    print("Model_Factors Table from PostgreSQL again:\n", model_factors_df)
