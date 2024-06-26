# From Python to Tidy R (and Back)
**A Running List of Key Python Operations Translated to (Mostly) Tidy R**

![Visitors](https://api.visitorbadge.io/api/visitors?path=https%3A%2F%2Fgithub.com%2Fpdwaggoner%2Fpython-to-tidy-R&label=Visitors&countColor=%ba68c8&style=plastic)

Frequently I am writing code in Python and R. And my team relies heavily on the [Tidyverse](https://www.tidyverse.org/) syntax. So, I am often translating key Python operations (pandas, matplotlib, etc.) to tidy R (dplyr, ggplot2, etc.). In an effort to ease that translation, and also to crowdsource a running directory of these translations, I created this repo. 

This is just a start. **Please feel free to share and also directly contribute or revise via pulls or issues**.

*Note:* I recommend using the native pipe operator (`|>`) when constructing piped operations in practice, instead of the `magrittr` pipe (`%>%`). However, I used the latter in this repo because the `|` in the native R pipe threw off formatting of the markdown tables. 

## Table of Contents
- [Key tasks](#Key-tasks)
- [Joining Data](#Joining-Data)
- [Iteration](#Iteration)
- [Iteration Over Lists](#Iteration-Over-Lists)
- [String Operations](#String-Operations)
- [Modeling and Machine Learning](#Modeling-and-Machine-Learning)
- [Network Modeling and Dynamics](#Network-Modeling-and-Dynamics)
- [Parallel Computing](https://github.com/pdwaggoner/python-to-tidy-R/blob/main/Parallel%20Computing.md)

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
| **Summary Statistics**  | `df.describe()`                      | `data %>% summary()` or `data %>% glimpse()`              |

## Joining Data

This is the only table that includes SQL given that most of the R/`dplyr` operations were patterned and named after many SQL operations.

| Join Type       | SQL                                      | Python (Pandas)                         | R (dplyr)                              |
|-----------------|------------------------------------------|----------------------------------------|----------------------------------------|
| **Inner Join**  | `INNER JOIN`                             | `pd.merge(df1, df2, on='key')`         | `inner_join(df1, df2, by='key')`       |
| **Left Join**   | `LEFT JOIN`                              | `pd.merge(df1, df2, on='key', how='left')` | `left_join(df1, df2, by='key')`        |
| **Right Join**  | `RIGHT JOIN`                             | `pd.merge(df1, df2, on='key', how='right')` | `right_join(df1, df2, by='key')`       |
| **Full Outer Join** | `FULL OUTER JOIN`                      | `pd.merge(df1, df2, on='key', how='outer')` | `full_join(df1, df2, by='key')`         |
| **Cross Join**  | `CROSS JOIN`                             | `pd.merge(df1, df2, how='cross')`       | Not directly supported, but can be achieved with `full_join` and filtering |
| **Anti Join**   | Not directly supported                   | `pd.merge(df1, df2, on='key', how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)` | Not directly supported, but can be achieved with `anti_join` function from dplyr or by using `filter()` and `!` condition |
| **Semi Join**   | Not directly supported                   | `pd.merge(df1, df2, on='key', how='inner', indicator=True).query('_merge == "both"').drop('_merge', axis=1)` | Not directly supported, but can be achieved with `semi_join` function from dplyr or by using `filter()` and `!` condition |
| **Self Join**   | `INNER JOIN` with the same table         | `pd.merge(df, df, on='key')`            | `inner_join(df, df, by='key')`          |
| **Multiple Key Join** | `INNER JOIN` with multiple keys     | `pd.merge(df1, df2, on=['key1', 'key2'])` | `inner_join(df1, df2, by=c('key1', 'key2'))` |
| **Join with Renamed Columns** | `INNER JOIN` with renamed columns | `pd.merge(df1.rename(columns={'col1': 'key'}), df2, on='key')` | `inner_join(rename(df1, key = col1), df2, by = 'key')` |
| **Join with Complex Condition** | `INNER JOIN` with complex conditions | `pd.merge(df1, df2, on='key', how='inner', left_on=(df1['col1'] > 10) & (df1['col2'] == df2['col3']))` | Not directly supported, but can be achieved with `filter()` and complex conditions |
| **Join with Different Key Names** | `INNER JOIN` with different key names | `pd.merge(df1, df2, left_on='key1', right_on='key2')` | `inner_join(df1, df2, by = c('key1' = 'key2'))` |

## Iteration

| Task / Operation            | Python (Pandas)                       | Tidyverse (dplyr and purrr)       |
|-----------------------------|--------------------------------------|-----------------------------------|
| **Iterate Over Rows**       | `for index, row in df.iterrows():`   | `data %>% rowwise() %>% mutate(new_col = your_function(col))` |
|                             | `    print(row['col1'], row['col2'])` |                                       |
| **Map Function to Column**  | `df['new_col'] = df['col'].apply(your_function)` | `data %>% mutate(new_col = map_dbl(col, your_function))` |
| **Apply Function to Column**| `df['new_col'] = your_function(df['col'])` | `data %>% mutate(new_col = your_function(col))` |
| **Group and Map**           | `for group, group_df in df.groupby('group_col'):` | `data %>% group_by(group_col) %>% nest(data = .) %>% mutate(new_col = map(data, your_function))` |
| **Map Over List Column**    | `df['new_col'] = df['list_col'].apply(lambda x: [your_function(i) for i in x])` | `data %>% mutate(new_col = map(list_col, ~map(your_function, .)))` |
| **Map with Anonymous Function** | - | `data %>% mutate(new_col = map_dbl(col, ~your_function(.)))` |
| **Map Multiple Columns**    | `df['new_col'] = df.apply(lambda row: your_function(row['col1'], row['col2']), axis=1)` | `data %>% mutate(new_col = pmap_dbl(list(col1, col2), ~your_function(...)))` |

## Iteration Over Lists

| Task / Operation                  | Python (Pandas)                          | Tidyverse (dplyr and purrr)               |
|-----------------------------------|-----------------------------------------|-------------------------------------------|
| **Map Function Across List Column**| `df['new_col'] = df['list_col'].apply(lambda x: [your_function(i) for i in x])` | `data %>% mutate(new_col = map(list_col, ~map(your_function, .)))` |
| **Nested Map in List Column**     | `df['new_col'] = df['list_col'].apply(lambda x: [your_function(i) for i in x])` | `data %>% mutate(new_col = map(list_col, ~map(your_function, .)))` |
| **Nested Map Across Columns**     | -                                       | `data %>% mutate(new_col = map2(list(col1, col2), ~map(your_function, .)))` |
| **Nested Map Within List Column** | -                                       | `data %>% mutate(new_col = map(list_col, ~map(your_function, .)))` |
| **Map Across Rows with Nested Map**| -                                     | `data %>% mutate(new_col = pmap(list(col1, col2), ~list(your_function(.x), your_function(.y))))` |
| **Nested Map Within Nested List**   | -                                       | `data %>% mutate(new_col = map(list(list_col), ~map(your_function, .)))` |
| **Nested Map Across List of Lists** | `df['new_col'] = df['list_col'].apply(lambda x: [list(map(your_function, i)) for i in x])` | `data %>% mutate(new_col = map2(list(list_col1, list_col2), ~map2(your_function1, your_function2, .x, .y)))` |
| **Nested Map Across Rows and Lists**| -                                     | `data %>% mutate(new_col = pmap(list(col1, col2, col3), ~list(your_function(.x), your_function(.y), your_function(.z))))` |
| **Map and Reduce Across List**      | `df['new_col'] = df['list_col'].apply(lambda x: reduce(your_function, x))` | `data %>% mutate(new_col = map(list_col, ~reduce(your_function, .)))` |
| **Map and Reduce Across Rows**      | `df['new_col'] = df.apply(lambda row: reduce(your_function, row[['col1', 'col2']]), axis=1)` | `data %>% mutate(new_col = pmap(list(col1, col2), ~reduce(your_function, .)))` |

## String Operations

| Task / Operation               | Python (Pandas)                    | Tidyverse (dplyr and stringr)            |
|--------------------------------|-----------------------------------|-----------------------------------------|
| **String Length**              | `df['col'].str.len()`             | `data %>% mutate(new_col = str_length(col))` |
| **Concatenate Strings**        | `df['new_col'] = df['col1'] + df['col2']` | `data %>% mutate(new_col = str_c(col1, col2))` |
| **Split Strings**              | `df['col'].str.split(', ')`      | `data %>% mutate(new_col = str_split(col, ', '))` |
| **Substring**                  | `df['col'].str.slice(0, 5)`      | `data %>% mutate(new_col = str_sub(col, 1, 5))` |
| **Replace Substring**          | `df['col'].str.replace('old', 'new')` | `data %>% mutate(new_col = str_replace(col, 'old', 'new'))` |
| **Uppercase / Lowercase**      | `df['col'].str.upper()`           | `data %>% mutate(new_col = str_to_upper(col))` |
|                               | `df['col'].str.lower()`           | `data %>% mutate(new_col = str_to_lower(col))` |
| **Strip Whitespace**           | `df['col'].str.strip()`           | `data %>% mutate(new_col = str_squish(col))` |
| **Check for Substring**        | `df['col'].str.contains('pattern')` | `data %>% mutate(new_col = str_detect(col, 'pattern'))` |
| **Count Substring Occurrences** | `df['col'].str.count('pattern')`  | `data %>% mutate(new_col = str_count(col, 'pattern'))` |
| **Find First Occurrence of Substring**| `df['col'].str.find('pattern')`        | `data %>% mutate(new_col = str_locate(col, 'pattern')[, 1])` |
| **Extract Substring with Regex**      | `df['col'].str.extract(r'(\d+)')`      | `data %>% mutate(new_col = str_extract(col, '(\\d+)'))` |
| **Remove Duplicates in Strings**      | -                                      | `data %>% mutate(new_col = str_unique(col))` |
| **Pad Strings**                       | `df['col'].str.pad(width=10, side='right', fillchar='0')` | `data %>% mutate(new_col = str_pad(col, width = 10, side = 'right', pad = '0'))` |
| **Truncate Strings**                  | `df['col'].str.slice(0, 10)`           | `data %>% mutate(new_col = str_sub(col, 1, 10))` |
| **Title Case**                        | -                                      | `data %>% mutate(new_col = str_to_title(col))` |
| **Join List of Strings**              | `'separator'.join(df['col'])`          | `data %>% mutate(new_col = str_flatten(col, collapse = 'separator'))` |
| **Remove Punctuation**                | -                                      | `data %>% mutate(new_col = str_remove_all(col, '[[:punct:]]'))` |
| **String Encoding/Decoding**          | -                                      | `data %>% mutate(new_col = str_encode(col, to = 'UTF-8'))` |

## Modeling and Machine Learning

| Task / Operation              | Python (scikit-learn)                   | R (various packages)                    |
|-------------------------------|----------------------------------------|----------------------------------------|
| **Data Preprocessing**        | `from sklearn.preprocessing import ...`  | `library(caret)`                       |
|                               | `from sklearn.pipeline import Pipeline` | `library(glmnet)`                      |
|                               | `preprocessor = ...`                  | `preprocess <- preProcess(data, ...)`   |
| **Feature Scaling**           | `StandardScaler()`                     | `preprocess$scaling`                    |
| **Feature Selection**         | `SelectKBest()`                        | `caret::createFolds()`                  |
| **Data Splitting**            | `train_test_split()`                   | `createDataPartition()`                 |
| **Model Initialization**      | `model = ...()`                        | `model <- ...()`                       |
| **Model Training**            | `model.fit(X_train, y_train)`          | `model <- train(y ~ ., data = data)`   |
| **Model Prediction**          | `y_pred = model.predict(X_test)`        | `y_pred <- predict(model, newdata)`    |
| **Model Evaluation**          | `accuracy_score(y_test, y_pred)`       | `confusionMatrix(y_pred, y_true)`      |
| **Hyperparameter Tuning**     | `GridSearchCV()`                       | `tuneGrid(...)`                        |
| **Cross-Validation**          | `cross_val_score()`                    | `trainControl(method = "cv")`           |
| **Model Pipelining**          | `pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])` | `model <- train(y ~ ., data = data, method = model, trControl = trainControl(method = "cv"))` |
| **Feature Engineering**         | `from sklearn.preprocessing import ...` | `library(caret)`                     |
|                                 | Custom feature transformers          | Custom feature transformers           |
| **Handling Missing Data**       | `SimpleImputer()`                     | `preprocess$impute`                   |
| **Encoding Categorical Data**   | `OneHotEncoder()`                     | `dummyVars()`                        |
| **Dimensionality Reduction**    | `PCA()`                               | `preprocess$reduce`                   |
| **Model Selection**             | `GridSearchCV()`                      | `caret::train()`                      |
| **Ensemble Learning**           | Various ensemble methods              | `caret::train()` with `method="stack"` |
| **Regularization**              | Lasso, Ridge, Elastic Net, etc.       | `glmnet()`                            |
| **Model Interpretability**      | SHAP, Lime, etc.                      | DALEX, iml, etc.                      |
| **Model Export/Serialization**   | `joblib` or `pickle`                  | `saveRDS` or other formats            |
| **Deploying Models**            | Web frameworks (e.g., Flask, Django)  | Web frameworks (e.g., Shiny, Plumber) |
| **Batch Scoring**               | Scripting or automation tools         | R batch processing                    |
| **Feature Scaling/Normalization**| `StandardScaler()`, `MinMaxScaler()`, etc. | `scale()`, `normalize()`, etc.       |
| **Feature Selection with L1 Regularization** | `SelectFromModel()`, `Lasso()`  | `glmnet()`, `cv.glmnet()`            |
| **Handling Imbalanced Data**    | `RandomUnderSampler()`, `SMOTE()`, etc. | `caret::train()` with `weights` or `sampling` |
| **Model Evaluation Metrics**    | `classification_report()`, `confusion_matrix()`, `mean_squared_error()`, etc. | `confusionMatrix()`, `postResample()`, `RMSE`, etc. |
| **Feature Importance**          | `.feature_importances_` (Random Forest, etc.) | `varImp()`, `vip()`, etc.         |
| **Model Persistence**           | `joblib`, `pickle`, `sklearn.externals` | `saveRDS`, `save()`, `serialize()`, etc. |
| **Time Series Forecasting**     | `Prophet`, `ARIMA`, `ExponentialSmoothing`, etc. | `forecast`, `prophet`, `auto.arima`, etc. |
| **Natural Language Processing (NLP)** | `nltk`, `spaCy`, `textblob`, etc. | `tm`, `quanteda`, `udpipe`, `tm.plugin.webmining`, etc. |
| **Deep Learning**               | `Keras`, `TensorFlow`, `PyTorch`, etc. | `keras`, `tensorflow`, `torch`, `mxnet`, etc. |
| **Model Interpretation**        | `SHAP`, `LIME`, `ELI5`, etc.         | `DALEX`, `iml`, `iBreakDown`, `lime`, etc. |
| **Model Deployment in Production** | Containers, cloud platforms (e.g., Docker, Kubernetes, AWS SageMaker) | Containers, Shiny, Plumber, APIs, cloud platforms |

## Network Modeling and Dynamics

| Task / Operation                | Python (NetworkX)                    | R (various packages)                    |
|---------------------------------|--------------------------------------|----------------------------------------|
| **Network Creation**            | `G = nx.Graph()`, `G.add_node()`, `G.add_edge()` | `igraph::graph()`, `add_vertices()`, `add_edges()` |
| **Node and Edge Attributes**    | `G.nodes[node]['attribute'] = value`, `G.edges[edge]['attribute'] = value` | `V(graph)$attribute <- value`, `E(graph)$attribute <- value` |
| **Network Visualization**       | `nx.draw(G)`, `matplotlib` for customization | `plot(graph)`, `igraph`, `ggplot2`, `visNetwork`, etc. |
| **Network Measures**            | `nx.degree_centrality(G)`, `nx.betweenness_centrality(G)`, `nx.clustering(G)`, etc. | `degree()`, `betweenness()`, `transitivity()`, etc. |
| **Community Detection**         | `community.detect()` (e.g., Louvain, Girvan-Newman) | `cluster_walktrap()`, `cluster_fast_greedy()`, `cluster_leading_eigen()`, etc. |
| **Link Prediction**             | `link_prediction.method()` (e.g., Common Neighbors, Jaccard Coefficient) | `link_prediction.method()` (e.g., Adamic-Adar, Preferential Attachment) |
| **Network Filtering/Selection** | `G.subgraph(nodes)`                | `subgraph(graph, vertices)`            |
| **Network Embedding**           | `node2vec`, `GraphSAGE`, etc.        | `walktrap.community`, `fastgreedy.community`, etc. |
| **Network Simulation**          | `nx.erdos_renyi_graph()`, `nx.watts_strogatz_graph()`, etc. | `igraph::erdos.renyi.game()`, `igraph::watts.strogatz.game()`, etc. |
| **Network Analysis Pipelines**  | Custom pipelines using NetworkX, Pandas, and other libraries | Custom pipelines using igraph, dplyr, and other packages |
| **Dynamic Network Analysis**    | `dynetx` for dynamic networks       | `tsna` for temporal networks, `dyngraph` for dynamic graphs, etc. |
| **Geospatial Network Analysis** | `osmnx` for urban network analysis  | `stplanr` for transport planning, `spatnet` for spatial network analysis, etc. |
| **Network Modeling for Machine Learning** | Integration with scikit-learn, PyTorch, etc. | Integration with caret, glmnet, keras, etc. |
| **Community Visualization**      | Visualization of detected communities using network layouts | `igraph::plot.igraph()` with community coloring |
| **Path Analysis**               | Shortest paths, k-shortest paths, and all simple paths | `get.shortest.paths()`, `all.simple.paths()` |
| **Centrality Analysis**         | Closeness centrality, eigenvector centrality, Katz centrality, etc. | `closeness()`, `eigen_centrality()`, `katz_centrality()`, etc. |
| **Structural Role Analysis**    | Structural equivalence, equivalence-based roles | `structural_equivalence()`, `role_equiv()`, etc. |
| **Network Robustness Analysis**  | Network attack simulations, robustness metrics | `robustness()` function, `remove_vertices()`, etc. |
| **Temporal Network Analysis**   | Temporal networks, evolving networks | `dynnet` package for dynamic networks, temporal extensions of `igraph` functions |
| **Multiplex Network Analysis**  | Analyzing multiple layers of networks | `multiplex` package for multilayer networks, `mgm` package for multilayer graphical models |
| **Network Alignment**           | Aligning nodes in two or more networks | `netAlign` package for network alignment, `gmatch` package for graph matching |
| **Dynamic Community Detection**  | Detecting evolving communities over time | `dynCOMM` for dynamic community detection |
| **Network Generative Models**   | Generating networks from various models (e.g., ER, BA, etc.) | `igraph::sample_gnm()`, `igraph::sample_degseq()`, etc. |
| **Geospatial Network Analysis** | Geospatial network analysis and routing | `stplanr` for transport planning, `spatnet` for spatial network analysis, etc. |
| **Network Modeling for Machine Learning** | Integrating network data with machine learning libraries | Combining `igraph` or custom network features with caret, glmnet, keras, etc. |
