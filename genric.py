# # import pandas as pd
# # import sys
# # import os
# # from sqlalchemy import create_engine, text
# # import pickle

# # # Add the directory containing the app module to the Python path
# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# # # Now import SessionLocal
# # from app.db.session import SessionLocal

# # def get_engine():
# #     # Create an SQLAlchemy engine using the existing session
# #     return SessionLocal().bind

# # def get_model_names():
# #     engine = get_engine()
# #     query = """SELECT DISTINCT "ModelName" FROM "UTCL_Optimizer"."Model_Factors" """
# #     df_Master = pd.read_sql(query, engine)
# #     return df_Master["ModelName"].tolist()

# # def get_model_target(model_name):
# #     engine = get_engine()
# #     query = f"""SELECT "VariableName" FROM "UTCL_Optimizer"."Model_Factors" WHERE "ModelName" ='{model_name}' AND "VariableType" ='Target' """
# #     df_Master = pd.read_sql(query, engine)
# #     return df_Master['VariableName'].tolist()

# # def get_state_variables(model_name):
# #     engine = get_engine()
# #     query = f"""SELECT "VariableName" FROM "UTCL_Optimizer"."Model_Factors" WHERE "ModelName" ='{model_name}' AND "VariableType" ='StateVariables' """
# #     df_Master = pd.read_sql(query, engine)
# #     return df_Master['VariableName'].tolist()

# # def get_positive_controls(model_name):
# #     engine = get_engine()
# #     query = f"""SELECT "VariableName" FROM "UTCL_Optimizer"."Model_Factors" WHERE "ModelName" ='{model_name}' AND "VariableType" ='PositiveControlVariables' """
# #     df_Master = pd.read_sql(query, engine)
# #     return df_Master['VariableName'].tolist()

# # def get_negative_controls(model_name):
# #     engine = get_engine()
# #     query = f"""SELECT "VariableName" FROM "UTCL_Optimizer"."Model_Factors" WHERE "ModelName" ='{model_name}' AND "VariableType" ='NegativeControlVariables' """
# #     df_Master = pd.read_sql(query, engine)
# #     return df_Master['VariableName'].tolist()

# # def get_all_variables(model_name):
# #     engine = get_engine()
# #     query = f"""SELECT DISTINCT "VariableName" FROM "UTCL_Optimizer"."Model_Factors" WHERE "ModelName" ='{model_name}' """
# #     df_Master = pd.read_sql(query, engine)
# #     return df_Master['VariableName'].tolist()

# # def get_process_lags(model_name):
# #     engine = get_engine()
# #     query = f"""SELECT "VariableName", "Process_Lag" FROM "UTCL_Optimizer"."Model_Factors" WHERE "ModelName" ='{model_name}' """
# #     df_Master = pd.read_sql(query, engine)
# #     return df_Master

# # def get_variable_limits(model_name):
# #     engine = get_engine()
# #     query = f"""SELECT "VariableName", "LL", "UL" FROM "UTCL_Optimizer"."Model_Factors" WHERE "ModelName" ='{model_name}' """
# #     df_Master = pd.read_sql(query, engine)
# #     return df_Master

# # def get_raw_data(variables: list):
# #     engine = get_engine()
# #     query = f"""
# #     SELECT DISTINCT "OPC_Tag", "Readable_Tag" FROM "UTCL_Optimizer"."T_Tag_Master" WHERE "OPC_Tag" IS NOT NULL AND "Readable_Tag" IS NOT NULL
# #     AND "Readable_Tag" IN {tuple(variables)}
# #     """
# #     df_Master = pd.read_sql(query, engine)

# #     tag_dict = {}
# #     for i in range(df_Master.shape[0]):
# #         tag_dict[df_Master['OPC_Tag'][i]] = df_Master['Readable_Tag'][i]

# #     # Get the list of columns from the table
# #     table_columns_query = """
# #     SELECT column_name
# #     FROM information_schema.columns
# #     WHERE table_schema = 'UTCL_Optimizer'
# #     AND table_name = 'combined_data'
# #     """
# #     table_columns_df = pd.read_sql(table_columns_query, engine)
# #     table_columns = table_columns_df['column_name'].tolist()

# #     required_columns = []
# #     for x in tag_dict.keys():
# #         if x is not None:
# #             required_columns.append(x)

# #     # Construct the query to include all required columns, using NULL for missing columns
# #     columns = ', '.join([f'"{col}"' if col in table_columns else f'NULL AS "{col}"' for col in required_columns])
# #     query = f"""
# #     SELECT "Time", {columns} FROM "UTCL_Optimizer".combined_data ORDER BY "Time" ASC 
# #     """
# #     df_TS = pd.read_sql(query, engine)
# #     # Convert the time column to datetime
# #     df_TS['Time'] = pd.to_datetime(df_TS['Time'])
# #     df_TS.set_index('Time', inplace=True)
# #     df_TS.rename(columns=tag_dict, inplace=True)
# #     return df_TS
# # def get_BP_span(model_name):
# #     engine = get_engine()
# #     query = f"""SELECT "VariableName", "BP", "SmoothingWindow" FROM "UTCL_Optimizer"."Model_Factors" WHERE "ModelName" ='{model_name}' """
# #     df_Master = pd.read_sql(query, engine)
# #     return df_Master

# # def load_model(model_name):
# #     read_query = f"""
# #     SELECT "Model" FROM "UTCL_Optimizer"."T_models" WHERE "ModelName" = '{model_name}'
# #     ORDER BY "TimeStamp" DESC
# #     LIMIT 1
# #     """
# #     session = SessionLocal()
# #     try:
# #         result = session.execute(text(read_query)).fetchone()
# #         if result:
# #             latest_model_pickle = result[0]
# #             model = pickle.loads(latest_model_pickle)
# #             return model
# #     except Exception as e:
# #         print(f"Error: {e}")
# #     finally:
# #         session.close() 



import pandas as pd
import sys
import os
from sqlalchemy import create_engine, text
import pickle

# Add the directory containing the app module to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import SessionLocal from session
from app.db.session import SessionLocal

def get_engine():
    # Create an SQLAlchemy engine using the existing session
    return SessionLocal().bind

def get_model_names():
    engine = get_engine()
    query = """SELECT DISTINCT "ModelName" FROM "UTCL_Optimizer"."Model_Factors" """
    df_models = pd.read_sql(query, engine)
    get_model_names = df_models["ModelName"].tolist()
    print("Model Names:", model_names)
    return model_names

def get_variables_by_type(model_name, variable_type):
    """Retrieve variables based on their type for a given model name."""
    engine = get_engine()
    query = f"""
    SELECT "VariableName" FROM "UTCL_Optimizer"."Model_Factors"
    WHERE "ModelName" = '{model_name}' AND "VariableType" = '{variable_type}'
    """
    df_vars = pd.read_sql(query, engine)
    variables = df_vars['VariableName'].tolist()
    print(f"{variable_type} variables for {model_name}:", variables)
    return variables

def get_all_variables(model_name):
    """Retrieve all variables for a given model name."""
    engine = get_engine()
    query = f"""
    SELECT DISTINCT "VariableName" FROM "UTCL_Optimizer"."Model_Factors"
    WHERE "ModelName" = '{model_name}'
    """
    df_vars = pd.read_sql(query, engine)
    all_vars = df_vars['VariableName'].tolist()
    print(f"All variables for {model_name}:", all_vars)
    return all_vars

def get_raw_data(variables):
    engine = get_engine()
    query = f"""
    SELECT DISTINCT "OPC_Tag", "Readable_Tag" FROM "UTCL_Optimizer"."T_Tag_Master"
    WHERE "OPC_Tag" IS NOT NULL AND "Readable_Tag" IS NOT NULL
    AND "Readable_Tag" IN {tuple(variables)}
    """
    df_tags = pd.read_sql(query, engine)
    tag_dict = {df_tags['OPC_Tag'][i]: df_tags['Readable_Tag'][i] for i in range(df_tags.shape[0])}
    print("Tag dictionary:", tag_dict)

    # Get the list of columns from the table
    table_columns_query = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = 'UTCL_Optimizer'
    AND table_name = 'combined_data'
    """
    table_columns_df = pd.read_sql(table_columns_query, engine)
    table_columns = table_columns_df['column_name'].tolist()
    print("Available columns in combined_data:", table_columns)

    required_columns = [x for x in tag_dict.keys() if x in table_columns]
    columns = ', '.join([f'"{col}"' if col in table_columns else f'NULL AS "{col}"' for col in required_columns])
    query = f"""
    SELECT "Time", {columns} FROM "UTCL_Optimizer".combined_data ORDER BY "Time" ASC
    """
    df_TS = pd.read_sql(query, engine)
    df_TS['Time'] = pd.to_datetime(df_TS['Time'])
    df_TS.set_index('Time', inplace=True)
    df_TS.rename(columns=tag_dict, inplace=True)
    print("Time series data fetched:", df_TS.head())
    return df_TS

def load_model(model_name):
    read_query = f"""
    SELECT "Model" FROM "UTCL_Optimizer"."T_models" WHERE "ModelName" = '{model_name}'
    ORDER BY "TimeStamp" DESC
    LIMIT 1
    """
    session = SessionLocal()
    try:
        result = session.execute(text(read_query)).fetchone()
        if result:
            latest_model_pickle = result[0]
            model = pickle.loads(latest_model_pickle)
            print(f"Model '{model_name}' loaded successfully.")
            return model
        else:
            print(f"No model found for '{model_name}'.")
    except Exception as e:
        print(f"Error loading model for '{model_name}': {e}")
    finally:
        session.close()

# Example usage
if __name__ == "__main__":
    model_name = "Kiln_Torque_Model"  # replace with your model name
    model_names = get_model_names()
    get_variables_by_type(model_name, "Target")         # Get target variables
    get_variables_by_type(model_name, "StateVariables") # Get state variables
    get_variables_by_type(model_name, "PositiveControlVariables") # Get positive controls
    get_variables_by_type(model_name, "NegativeControlVariables") # Get negative controls
    get_all_variables(model_name)
    load_model(model_name)
