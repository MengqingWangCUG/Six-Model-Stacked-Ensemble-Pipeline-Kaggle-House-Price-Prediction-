# Advanced Regression Framework: A Hybrid Ensemble Approach with Deep Diagnostics

This is a solution developed for the Kaggle House Prices: Advanced Regression Techniques training project. Beyond merely optimizing a competition metric, the proposed architecture serves as a generalizable framework designed for complex tabular regression tasks.
The core philosophy of this solution anchors on robustness through ensemble diversity. By integrating regularized linear models (Lasso, Ridge, ElasticNet) with Gradient Boosted Decision Trees (LightGBM, XGBoost, CatBoost) through a multi-stage stacking strategy, the system effectively captures both broad linear trends and intricate non-linear feature interactions while minimizing overfitting. The current score stands at 0.12181, representing top-tier performance among publicly available notebooks.The model's current performance is constrained by the limited size of the training dataset. However, I am confident that if this architecture were extended to tasks with more substantial data availability, it would achieve significantly better performance.

The pipeline consists of four distinct phases: Advanced Preprocessing, Hybrid Modeling, Bayesian Optimization, and Hierarchical Ensembling.
https://www.kaggle.com/code/mengqingcncug/wmq-kaggle-advanced-regression-techniques

## Feature Engineering
Feature Engineering(preprocess)comprises four core mechanisms:
Hierarchical & Informative Imputation
Spatially correlated missingness (e.g., LotFrontage) is reconstructed via neighborhood-level medians, while structural sparsity (e.g., PoolQC, GarageType) is handled by introducing explicit "Absence Indicators" (binary flags) to capture the informative value of missing data. Distributional Rectification via Power TransformsTo satisfy the homoscedasticity assumptions of linear estimators, the pipeline dynamically assesses feature skewness. Variables exceeding a skewness threshold of $|0.75|$ undergo non-linear transformations, automatically selecting between Box-Cox and Yeo-Johnson to maximize normality.Leakage-Resistant Target EncodingFor high-cardinality categorical features, the system implements K-Fold Regularized Target Encoding. By computing category means within isolated cross-validation folds, the architecture explicitly prevents target leakage.
the pipeline engineers "holistic" meta-attributes to capture interaction effects. This includes aggregating surface areas, synthesizing quality-condition indices, and converting sparse continuous variables into binary density flags, thereby enhancing the decision boundaries for tree-based ensembles.



## The Hybrid Modeling Strategy
The framework splits the input data into two streams to maximize the strengths of different algorithm families:
1. Linear Stream :
Algorithms including Lasso, Ridge, ElasticNet. To capture linear dependencies and extrapolate trends. The use of RobustScaler ensures outliers do not disproportionately affect the coefficients. Progressively granular alpha grids to ensure convergence.Given the nature of the competition dataset, this part appears to be the primary driver of the model's predictive performance.

2. Non-Linear Stream:
Using LightGBM, XGBoost and CatBoost to capture complex, non-linear interactions and threshold-based patterns.Hyperparameters are tuned using Bayesian Optimization via the Tree-Structured Parzen Estimator sampler to find the global optimum efficiently.

3. Hierarchical Ensemble Mechanism
Rather than a simple average, the solution employs a stacking and blending architecture.
Step 1: The predictions from the 6 base models are generated using 5-Fold Cross-Validation. This creates OOF predictions for the training set, serving as meta-features for the next layer.
Step 2: A meta-model is trained on the meta-features. Uniquely, this framework enhances the meta-features by adding:
Step 3: The final prediction is a weighted blend of the Stacking Output  and the Optimization Output . This "ensemble of ensembles" ensures that neither the linear meta-model nor the static weights dominate, providing a highly stable prediction.

