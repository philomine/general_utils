import numpy as np
import pandas as pd
import pyspark.sql.functions as f
import pyspark.sql.types as t


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
    str_cols = dtypes[np.array(list(map(str, dtypes.values))) == "object"].keys()
    for col in str_cols:
        pd_df[col] = pd_df[col].astype(str)
    return pd_df


def spark_schema(spark_df):
    return pd.DataFrame(
        [[field.name, str(field.dataType), field.nullable] for field in spark_df.schema],
        columns=["name", "data_type", "nullable"],
    )


def spark_replace(spark_df, old_value, new_value, columns=None):
    """Replaces a value by another in a spark df. Set columns to a list of
    columns to replace only in those columns. Careful: the type of the columns
    will be kept, so if you want to replace 1 with "foo", you should cast the
    corresponding column to StringType() beforehand.

    Parameters
    ----------
    spark_df: pyspark dataframe
        The df in which to replace the value
    old_value: string, number or None
        The value to replace, set to None to replace null values (equivalent
        to nafill)
    new_value: string, number or None
        The value with which you will replace
    columns: array-like of string (optional, default: None)
        If you don't want to replace the values everywhere, give a list of
        columns in which you want to replace. If set to None, the value is
        looked for and replaced in every column.

    Returns
    -------
    spark_df: pyspark dataframe
        The modified df
    """
    if columns is None:
        for col in spark_df.columns:
            spark_df = spark_replace_in_col(spark_df, col, old_value, new_value)
    else:
        for col in columns:
            spark_df = spark_replace_in_col(spark_df, col, old_value, new_value)
    return spark_df


def spark_replace_in_col(spark_df, column_name, old_value, new_value):
    """Replaces a value with another in a given column of a spark df. Careful:
    the type of the columns will be kept, so if you want to replace 1 with
    "foo", you should cast the corresponding column to StringType() beforehand.

    Parameters
    ----------
    spark_df: pyspark dataframe
        The df in which to replace the value
    column_name: string
        The name of the column in which you want to replace the value, should
        be in spark_df.columns
    old_value: string, number or None
        The value to replace, set to None to replace null values (equivalent
        to nafill)
    new_value: string, number or None
        The value with which you will replace

    Returns
    -------
    spark_df: pyspark dataframe
        The modified df
    """
    dtype = {col: dtype for col, dtype in spark_df.dtypes}[column_name]
    if old_value is None:
        spark_df = spark_df.withColumn(
            column_name,
            f.when(f.col(column_name).isNull(), f.lit(new_value)).otherwise(f.col(column_name)).cast(dtype),
        )
    else:
        spark_df = spark_df.withColumn(
            column_name,
            f.when(f.col(column_name) == old_value, f.lit(new_value)).otherwise(f.col(column_name)).cast(dtype),
        )
    return spark_df


def spark_drop_empty_columns(spark_df):
    nb_fill = spark_df
    for col in nb_fill.columns:
        nb_fill = nb_fill.withColumn(col, f.col(col).isNotNull().cast("integer"))
    nb_fill = nb_fill.agg({col: "sum" for col in nb_fill.columns}).toPandas()
    nb_fill.columns = [col[4:-1] for col in nb_fill.columns]
    nb_fill = nb_fill.T
    nb_fill.columns = ["nb_fill"]

    empty_columns = nb_fill[nb_fill.nb_fill == 0].index.values
    for col in empty_columns:
        spark_df = spark_df.drop(col)

    return spark_df


def spark_union(dataframes):
    """Unions dataframes in dataframes list."""
    table = dataframes[0]
    for df in dataframes[1:]:
        df1 = table
        df2 = df

        new_columns_1 = [col for col in df2.columns if col not in df1.columns]
        new_columns_2 = [col for col in df1.columns if col not in df2.columns]
        for column in new_columns_1:
            df1 = df1.withColumn(column, f.lit(None))
        for column in new_columns_2:
            df2 = df2.withColumn(column, f.lit(None))
        table = df1.unionByName(df2)
    return table


def spark_any(spark_df, new_col, columns):
    spark_df = spark_df.withColumn(new_col, f.lit(False))
    for col in columns:
        spark_df = spark_df.withColumn(new_col, f.when(f.col(new_col) | f.col(col), True).otherwise(f.lit(False)))
    return spark_df


def spark_all(spark_df, new_col, columns):
    spark_df = spark_df.withColumn(new_col, f.lit(True))
    for col in columns:
        spark_df = spark_df.withColumn(new_col, f.when(f.col(new_col) & f.col(col), True).otherwise(f.lit(False)))
    return spark_df


def one_row_per_day(
    df, id_cols, start_date_col, end_date_col, include_start_date=True, include_end_date=True, return_start_end=False
):
    """Takes in a pyspark dataframe with id_cols and start and end dates, and
    creates one row per day per period.
    """
    if start_date_col == "date":
        df = df.withColumnRenamed(start_date_col, "start_date")
        start_date_col = "start_date"
    if end_date_col == "date":
        df = df.withColumnRenamed(end_date_col, "end_date")
        end_date_col = "end_date"

    # Rows should have every value and should be identifiable by id_cols + start_date_col
    df = df.select(*id_cols, start_date_col, end_date_col).dropna()
    df = df.groupby(id_cols + [start_date_col]).agg(f.max(f.col(end_date_col)).alias(end_date_col))

    temp = (
        df.select(*id_cols, start_date_col, end_date_col)
        .drop_duplicates()
        .withColumn("duration", f.datediff(end_date_col, start_date_col))
        .withColumn("duration_in_comma", f.expr("split(repeat(',', duration), ',')"))
        .select("*", f.posexplode("duration_in_comma").alias("date", "val"))
        .drop("duration_in_comma", "val", "duration")
        .withColumn("date", f.expr(f"date_add({start_date_col}, date)"))
    )

    if not include_start_date:
        temp = temp.filter(f.col("date") != f.col(start_date_col))
    if not include_end_date:
        temp = temp.filter(f.col("date") != f.col(end_date_col))
    if not return_start_end:
        temp = temp.drop(start_date_col, end_date_col)

    return temp


def spark_schema_from_dict(schema):
    spark_types = {
        "string": t.StringType(),
        "long": t.LongType(),
        "datetime": t.TimestampType(),
        "timestamp": t.TimestampType(),
        "double": t.DoubleType(),
        "bool": t.BooleanType(),
        "boolean": t.BooleanType(),
        "date": t.DateType(),
        "float": t.FloatType(),
        "int": t.IntegerType(),
        "integer": t.IntegerType(),
        "decimal(38,18)": t.DecimalType(38, 18),
    }

    return_schema = []
    for field, field_type in schema.items():
        field_type = spark_types[field_type]
        return_schema.append(t.StructField(field, field_type, True))
    return_schema = t.StructType(return_schema)

    return return_schema
