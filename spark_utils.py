import numpy as np
import pandas as pd
import pyspark.sql.functions as F


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


def spark_schema(spark_df):
    return pd.DataFrame(
        [
            [field.name, str(field.dataType), field.nullable]
            for field in spark_df.schema
        ],
        columns=["name", "data_type", "nullable"],
    )


def spark_replace(spark_df, column_name, old_value, new_value):
    return spark_df.withColumn(
        column_name,
        F.when(F.col(column_name) == old_value, F.lit(new_value)).otherwise(
            F.col(column_name)
        ),
    )


def spark_drop_empty_columns(spark_df):
    nb_fill = spark_df
    for col in nb_fill.columns:
        nb_fill = nb_fill.withColumn(
            col, F.col(col).isNotNull().cast("integer")
        )
    nb_fill = nb_fill.agg({col: "sum" for col in nb_fill.columns}).toPandas()
    nb_fill.columns = [col[4:-1] for col in nb_fill.columns]
    nb_fill = nb_fill.T
    nb_fill.columns = ["nb_fill"]

    empty_columns = nb_fill[nb_fill.nb_fill == 0].index.values
    for col in empty_columns:
        spark_df = spark_df.drop(col)

    return spark_df
