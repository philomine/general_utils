# General utils package

The general utils package distributes useful modules in data analysis,
vizualisation, machine learning, reporting, etc.

## List of modules

- data_analysis
  - function get_string_format
  - function get_dist_table
- ml_logger
  - class MLLogger
    - function log
- pandas_extensions
/!\ `import pandas_extensions` runs the extensions, no need to import the
functions
  - function pd.DataFrame.drop_sparse_columns
  - property pd.DataFrame.full_row_percentage
  - function pd.DataFrame.dropna_analysis
  - function pd.DataFrame.column_analysis
  - property pd.Series.is_id
  - property pd.DataFrame.is_id
  - function pd.DataFrame.divide_dataset
  - property pd.Series.nan_figure
  - function pd.DataFrame.plotly_report
  - function pd.Series.remove_outliers
  - function pd.DataFrame.plot_distribution
- plotly_reporter
  - function generate_report
- stringlist_utils
  - function stringlist_length
  - function stringlist_unique
  - function translate_stringlist
  - function append_stringlists
  - function stringlists_dist
- tools
  - function clear_terminal
  - function parent_dir
  - function elapsed_time
  - function partition_n_into_k_subsets
- vizualisation
  - function sample_pie_chart
  - function dist_table_pie_chart
  - function sample_bar_chart
  - function dist_table_bar_chart
  - function scatter_plot
  - function time_series_distribution
  - function numeric_distribution
  - function text_distribution

## Documentation

All functions and classes have a doctring. However, the vizualisation API might
need some extra explanation. Several viz functions interact in the following
way:

```
pd.plot_distribution
|-- numeric_distribution
|-- time_series_distribution
|-- text_distribution
    |-- sample_pie_chart
    |   |-- dist_table_pie_chart
    |-- sample_bar_chart
        |-- dist_table_bar_chart
```

You can call any of these functions independantly. However, calling one of
these will call all underlying functions.
