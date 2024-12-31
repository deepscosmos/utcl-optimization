import pandas as pd
import sys
import os

# Add the directory containing the app module to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.db.session import SessionLocal


# def get_model_names():
#     conn = SessionLocal().bind.raw_connection()
#     query = """select distinct "ModelName" from "UTCL_Optimizer"."Model_Factors" """
#     df_Master = pd.read_sql(query, conn)
#     return df_Master["ModelName"].tolist()
def get_model_names():
    conn = SessionLocal().bind.raw_connection()
    query = """SELECT DISTINCT "ModelName" FROM "UTCL_Optimizer"."Model_Factors";"""
    print(f"Executing Query: {query}")  # Debug the query
    df_Master = pd.read_sql(query, conn)
    print(f"Query Result: {df_Master}")  # Debug the result
    return df_Master["ModelName"].tolist()



def get_model_target(model_name):
    conn = SessionLocal().bind.raw_connection()
    query = f"""select "VariableName" from "UTCL_Optimizer"."Model_Factors" 
                where "ModelName" ='{model_name}' and "VariableType" ='Target' """
    df_Master = pd.read_sql(query, conn)
    return df_Master['VariableName'].tolist()


def get_state_variables(model_name):
    conn = SessionLocal().bind.raw_connection()
    query = f"""select "VariableName" from "UTCL_Optimizer"."Model_Factors" 
                where "ModelName" ='{model_name}' and "VariableType" ='StateVariables' """
    df_Master = pd.read_sql(query, conn)
    return df_Master['VariableName'].tolist()


def get_positive_controls(model_name):
    conn = SessionLocal().bind.raw_connection()
    query = f"""select "VariableName" from "UTCL_Optimizer"."Model_Factors" 
                where "ModelName" ='{model_name}' and "VariableType" ='PositiveControlVariables' """
    df_Master = pd.read_sql(query, conn)
    return df_Master['VariableName'].tolist()


def get_negative_controls(model_name):
    conn = SessionLocal().bind.raw_connection()
    query = f"""select "VariableName" from "UTCL_Optimizer"."Model_Factors" 
                where "ModelName" ='{model_name}' and "VariableType" ='NegativeControlVariables' """
    df_Master = pd.read_sql(query, conn)
    return df_Master['VariableName'].tolist()


def get_all_variables(model_name):
    conn = SessionLocal().bind.raw_connection()
    query = f"""select distinct "VariableName" from "UTCL_Optimizer"."Model_Factors" 
                where "ModelName" ='{model_name}' """
    df_Master = pd.read_sql(query, conn)
    return df_Master['VariableName'].tolist()


def get_raw_data(variables: list):
    conn = SessionLocal().bind.raw_connection()
    query = f"""
    select distinct "OPC_Tag","Readable_Tag" from "UTCL_Optimizer"."T_Tag_Master" 
    where "OPC_Tag" is not null and "Readable_Tag" is not null
    and "Readable_Tag" in {tuple(variables)}
    """
    df_Master = pd.read_sql(query, conn)

    tag_dict = {}
    for i in range(df_Master.shape[0]):
        tag_dict[df_Master['OPC_Tag'][i]] = df_Master['Readable_Tag'][i]

    # Get the list of columns from the table
    table_columns_query = """
    SELECT column_name
    FROM information_schema.columns
    WHERE table_schema = 'UTCL_Optimizer'
    AND table_name = 'combined_data'
    """
    table_columns_df = pd.read_sql(table_columns_query, conn)
    table_columns = table_columns_df['column_name'].tolist()

    required_columns = []
    for x in tag_dict.keys():
        if x is not None:
            required_columns.append(x)

    # Construct the query to include all required columns, using NULL for missing columns
    columns = ', '.join([f'"{col}"' if col in table_columns else f'NULL AS "{col}"' for col in required_columns])
    query = f"""
    select "Time", {columns} from "UTCL_Optimizer".combined_data order by "Time" asc 
    """
    df_TS = pd.read_sql(query, conn)

    # Convert the time column to datetime
    df_TS['Time'] = pd.to_datetime(df_TS['Time'])
    df_TS.set_index('Time', inplace=True)
    df_TS.rename(columns=tag_dict, inplace=True)
    return df_TS


# === Main Script to Print All Results ===
if __name__ == "__main__":
    # Get and print all model names
    print("Fetching Model Names...")
    model_names = get_model_names()
    print(f"Model Names: {model_names}\n")

    # Iterate over each model name and fetch related information
    for model_name in model_names:
        print(f"Fetching details for model: {model_name}\n")

        # Get and print target variables
        targets = get_model_target(model_name)
        print(f"Target Variables: {targets}")

        # Get and print state variables
        state_variables = get_state_variables(model_name)
        print(f"State Variables: {state_variables}")

        # Get and print positive control variables
        positive_controls = get_positive_controls(model_name)
        print(f"Positive Control Variables: {positive_controls}")

        # Get and print negative control variables
        negative_controls = get_negative_controls(model_name)
        print(f"Negative Control Variables: {negative_controls}")

        # Get and print all variables
        all_variables = get_all_variables(model_name)
        print(f"All Variables: {all_variables}")

        # Get and print raw data for variables (if any)
        if all_variables:
            print(f"Fetching raw data for variables: {all_variables}")
            raw_data = get_raw_data(all_variables)
            print(f"Raw Data:\n{raw_data}\n")
        else:
            print("No variables found for raw data.\n")
