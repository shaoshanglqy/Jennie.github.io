---
title: Explain Model with SHAP Value
date: 2020-01-05 10:56:51
tags: ['Models','Python']
categories: ['Models']
---

### Import related libriaires and define variables

{% code lang:python %}
    import pandas as pd
    import numpy as np
    np.random.seed(0)
    import matplotlib.pyplot as plt
    df = pd.read_csv('/winequality-red.csv') # Load the data
    from sklearn.model_selection import train_test_split
    from sklearn import preprocessing
    from sklearn.ensemble import RandomForestRegressor
    # The target variable is 'quality'.
    Y = df['quality']
    X =  df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar','chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density','pH', 'sulphates', 'alcohol']]
    # Split the data into train and test data:
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
    # Build the model with the random forest regression algorithm:
    model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
    model.fit(X_train, Y_train)
{% endcode %}
<!-- more -->

### Plot Feature Importance with Blue Bar Charts

{% code lang:python %}
    import shap
    shap_values = shap.TreeExplainer(model).shap_values(X_train)
    shap.summary_plot(shap_values, X_train, plot_type="bar")
{% endcode %}


### Plot the Red&Blue Feature Importance Plot

{% code lang:python %}
    shap.summary_plot(shap_values, X_train)
{% endcode %}


### Partial Dependence Plot
The partial dependence plot shows the marginal effect one or two features have on the predicted outcome of a machine learning model

{% code lang:python %}
    shap.dependence_plot(“alcohol”, shap_values, X_train)
{% endcode %}

View the original article on Mediumn:
[Explain Your Model with the SHAP Values](https://towardsdatascience.com/explain-your-model-with-the-shap-values-bc36aac4de3d)