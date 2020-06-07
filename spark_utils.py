import pandas as pd


def spark_clean_df(pd_df):
    # First thing is to clean column names
    spa_illegal_characters = [" ", ",", "-", ";", "\n", "\t", "="]
    del_illegal_characters = [".", "{", "}", "(", ")"]
    new_columns = []
    for column in pd_df.columns:
        column = column.lower()
        for character in spa_illegal_characters:
            column = column.replace(character, "_")
        for character in del_illegal_characters:
            column = column.replace(character, "")
        new_columns.append(column)
    pd_df.columns = new_columns

    # Then, we cast object columns as strings
    dtypes = pd_df.dtypes
    str_cols = dtypes[
        np.array(list(map(str, dtypes.values))) == "object"
    ].keys()
    for col in str_cols:
        pd_df[col] = pd_df[col].astype(str)
    return pd_df
