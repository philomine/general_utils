import numpy as np
import pandas as pd
import pyspark
import pyspark.sql.functions as F


def replace(self, old_value, new_value, columns=None):
    """ Replaces a value by another in a spark df. Set columns to a list of 
    columns to replace only in those columns. Careful: the type of the columns
    will be kept, so if you want to replace 1 with "foo", you should cast the 
    corresponding column to StringType() beforehand.

    Parameters
    ----------
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
    spark_df = self
    if columns is None:
        for col in spark_df.columns:
            try:
                spark_df = spark_df.replace_in_col(col, old_value, new_value)
            except:
                pass
    else:
        for col in columns:
            try:
                spark_df = spark_df.replace_in_col(col, old_value, new_value)
            except:
                pass
    return spark_df


def replace_in_col(self, column_name, old_value, new_value):
    """ Replaces a value with another in a given column of a spark df. Careful: 
    the type of the columns will be kept, so if you want to replace 1 with 
    "foo", you should cast the corresponding column to StringType() beforehand. 
    
    Parameters
    ----------
    column_name: string
        The name of the column in which you want to replace the value, should 
        be in self.columns
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
    spark_df = self
    dtype = {col: dtype for col, dtype in spark_df.dtypes}[column_name]
    if old_value is None:
        spark_df = spark_df.withColumn(
            column_name,
            F.when(F.col(column_name).isNull(), F.lit(new_value)).otherwise(F.col(column_name)).cast(dtype),
        )
    else:
        spark_df = spark_df.withColumn(
            column_name,
            F.when(F.col(column_name) == old_value, F.lit(new_value)).otherwise(F.col(column_name)).cast(dtype),
        )
    return spark_df


setattr(pyspark.sql.dataframe.DataFrame, "replace", replace)
setattr(pyspark.sql.dataframe.DataFrame, "replace_in_col", replace_in_col)
