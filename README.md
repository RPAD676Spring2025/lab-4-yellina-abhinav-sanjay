[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/h0OTrYQA)
# Algorithms in Python

## Core Python Algorithm Libraries
### 1. NumPy
NumPy provides the foundation for scientific computing in Python is essential for algorithm implementation.

#### Examples:
import numpy as np

#### Create arrays for algorithm input
data = np.array([1,2,3,4,5])

#### Basic statistical operations
mean = np.mean(data)

std = np.std(data)

#### Matrix operations for algorithms
matrix_a = np.array([[1, 2], [3, 4]])

matrix_b = np.array([[5, 6], [7, 8]])

dot_product = np.dot(matrix_a, matrix_b)

### 2. SciPy
SciPy extends NumPy with additional scientific algorithms.

Useful modules for algorithm design:

-scipy.optimize: Optimization algorithms

-scipy.stats: Statistical distributions and tests

-scipy.sparse: Sparse matrix algorithms

-scipy.spatial: Spatial algorithms (e.g., k-d trees)

#### Examples:

from scipy.optimize import minimize

#### Define an objective function
def objective(x):
    return x[0]**2 + x[1]**2

#### Minimize the function
result = minimize(objective, [1, 1])
optimal_x = result.x

### 3. scikit-learn
Scikit-learn provides implementations of many standard machine learning algorithms with a consistent API.

Key algorithm categories:

-Classification (SVM, Random Forests, etc.)

-Regression (Linear, Logistic, etc.)

-Clustering (K-means, DBSCAN, etc.)

-Dimensionality reduction (PCA, t-SNE)

-Model selection and evaluation

#### Example:
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#### Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#### Train algorithm
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

#### Evaluate algorithm
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)


### 4. NetworkX
NetworkX is essential for graph-based algorithms and network analysis

Applications:

-Social network analysis

-Path finding algorithms

-Centrality measures

-Graph visualization

#### Example:
import networkx as nx
import matplotlib.pyplot as plt

#### Create a graph
G = nx.Graph()
G.add_edges_from([(1, 2), (1, 3), (2, 3), (3, 4)])

#### Run graph algorithms
shortest_path = nx.shortest_path(G, source=1, target=4)
centrality = nx.betweenness_centrality(G)

#### Visualize results
nx.draw(G, with_labels=True)
plt.show()


## Fairness-Aware Machine Learning Libraries

### 1. Fairlearn
Fairlearn is specifically designed to assess and improve the fairness of machine learning models.

Key features:

-Fairness metrics computation

-Mitigation algorithms for unfairness

-Visualization tools for fairness assessment

#### Examples

pip install fairlearn

from fairlearn.metrics import demographic_parity_difference
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

#### Evaluate fairness of a model
dpd = demographic_parity_difference(
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=sensitive_attributes_test
)

#### Mitigate unfairness
constraint = DemographicParity()
mitigator = ExponentiatedGradient(
    estimator=LogisticRegression(),
    constraints=constraint
)

mitigator.fit(
    X_train, y_train,
    sensitive_features=sensitive_attributes_train
)

fair_predictions = mitigator.predict(X_test)

### 2. AI Fairness 360 (AIF360)
Developed by IBM, AIF360 provides a comprehensive set of algorithms for detecting and mitigating bias. 

Key components:

-Pre-processing algorithms to transform biased data

-In-processing algorithms to incorporate fairness constraints

-Post-processing algorithms to adjust model outputs

#### Examples

pip install aif360

from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

#### Create a dataset with fairness information
dataset = BinaryLabelDataset(
    df=df,
    label_names=['outcome'],
    protected_attribute_names=['protected_attribute']
)

#### Measure bias
metric = BinaryLabelDatasetMetric(
    dataset,
    unprivileged_groups=[{'protected_attribute': 0}],
    privileged_groups=[{'protected_attribute': 1}]
)

#### Apply a bias mitigation algorithm
reweighing = Reweighing(
    unprivileged_groups=[{'protected_attribute': 0}],
    privileged_groups=[{'protected_attribute': 1}]
)

transformed_dataset = reweighing.fit_transform(dataset)

### 3. Aequitas
Aequitas is an open-source bias and fairness audit toolkit specifically designed for public policy applications.

Features:

-Group fairness metrics

-Bias reporting

-Visualization tools

#### Examples

pip install aequitas

from aequitas.preprocessing import preprocess_input_df
from aequitas.group import Group
from aequitas.bias import Bias

#### Preprocess the data
df_preprocessed = preprocess_input_df(df)

#### Group data and calculate group metrics
g = Group()
group_metrics = g.get_group_metrics(df_preprocessed)

#### Calculate bias metrics
b = Bias()
bias_metrics = b.get_disparity_predefined_groups(
    group_metrics,
    attribute_name='race',
    ref_group_value='white'
)

#### Print bias metrics
absolute_metrics = bias_metrics[0]
disparities = bias_metrics[1]

