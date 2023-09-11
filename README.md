# From Python to Tidy R (and back): 
## A Running List of Key Python Operations Translated to Tidy R

Frequrntly I am writing code in Python and R. And my team relies heavily on the tidyverse syntax. So, I am often translating key python operations (pandas, matplotlib, etc.) to tidy R (dplyr, ggplot2, etc.). In an effort to ease that translation, and also to crowdsoruce a running directory of these translations, I created this repo. 

This is just a start. **Please feel free to directly contirbute via pulls or issues**. Thanks!

----

## Key tasks

| Task / Operation         | Python (Pandas)                       | Tidyverse (dplyr, ggplot2)         |
|-------------------------|--------------------------------------|-----------------------------------|
| **Data Loading**        | `import pandas as pd`                | `library(readr)`                  |
|                         | `df = pd.read_csv('file.csv')`       | `data <- read_csv('file.csv')`    |
| **Select Columns**      | `df[['col1', 'col2']]`              | `data %>% select(col1, col2)`    |
| **Filter Rows**         | `df[df['col'] > 5]`                 | `data %>% filter(col > 5)`        |
| **Arrange Rows**        | `df.sort_values(by='col')`           | `data %>% arrange(col)`           |
| **Mutate (Add Columns)**| `df['new_col'] = df['col1'] + df['col2']` | `data %>% mutate(new_col = col1 + col2)` |
| **Group and Summarize** | `df.groupby('col').agg({'col2': 'mean'})` | `data %>% group_by(col) %>% summarize(mean_col2 = mean(col2))` |
| **Pivot/Wide to Long**  | `pd.melt(df, id_vars=['id'], var_name='variable', value_name='value')` | `data %>% gather(variable, value, -id)` |
| **Long to Wide/Pivot**  | `df.pivot(index='id', columns='variable', values='value')` | `data %>% spread(variable, value)` |
| **Data Visualization**  | Matplotlib, Seaborn, Plotly, etc.   | ggplot2                           |
|                         | `import matplotlib.pyplot as plt`   | `library(ggplot2)`                 |
|                         | `plt.scatter(df['x'], df['y'])`    | `ggplot(data, aes(x=x, y=y)) + geom_point()` |
| **Data Reshaping**      | `pd.concat([df1, df2], axis=0)`     | `bind_rows(df1, df2)`             |
|                         | `pd.concat([df1, df2], axis=1)`     | `bind_cols(df1, df2)`             |
| **String Manipulation** | `df['col'].str.replace('a', 'b')`   | `data %>% mutate(col = str_replace(col, 'a', 'b'))` |
| **Date and Time**      | `pd.to_datetime(df['date_col'])`    | `data %>% mutate(date_col = as.Date(date_col))` |
| **Missing Data Handling**| `df.dropna()`                        | `data %>% drop_na()`              |
| **Rename Columns**      | `df.rename(columns={'old_col': 'new_col'})` | `data %>% rename(new_col = old_col)` |
| **Summary Statistics**  | `df.describe()`                      | `data %>% summary()`              |

## Iteration

| Task / Operation            | Python (Pandas)                       | Tidyverse (dplyr and purrr)       |
|-----------------------------|--------------------------------------|-----------------------------------|
| **Iterate Over Rows**       | `for index, row in df.iterrows():`   | `data %>% rowwise() %>% mutate(new_col = your_function(col))` |
|                             | `    print(row['col1'], row['col2'])` |                                       |
| **Map Function to Column**  | `df['new_col'] = df['col'].apply(your_function)` | `data |> mutate(new_col = map_dbl(col, your_function))` |
| **Apply Function to Column**| `df['new_col'] = your_function(df['col'])` | `data |> mutate(new_col = your_function(col))` |
| **Group and Map**           | `for group, group_df in df.groupby('group_col'):` | `data |> group_by(group_col) |> nest(data = .) |> mutate(new_col = map(data, your_function))` |
| **Map Over List Column**    | `df['new_col'] = df['list_col'].apply(lambda x: [your_function(i) for i in x])` | `data |> mutate(new_col = map(list_col, ~map(your_function, .)))` |
| **Map with Anonymous Function** | - | `data |> mutate(new_col = map_dbl(col, ~your_function(.)))` |
| **Map Multiple Columns**    | `df['new_col'] = df.apply(lambda row: your_function(row['col1'], row['col2']), axis=1)` | `data |> mutate(new_col = pmap_dbl(list(col1, col2), ~your_function(...)))` |
