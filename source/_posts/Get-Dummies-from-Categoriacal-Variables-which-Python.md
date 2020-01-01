---
title: Get Dummies from Categoriacal Variables which Python
date: 2020-01-01 23:35:56
tags: 'Python'
categories: 'Python'
---

{% note %}

## Notes

- Get the categorical vars you want to get dummy with.
- use pd.get_dummies to convert one categorical variables to several dummys
{% endnote %}

{% code lang:python %}
catagorical_vars = ['peak','business_line','gender','age_level']
continuous_vars = set.difference(set(all_vars),set(catagorical_vars))

cate_list = []
for i in catagorical_vars:
    print(catagorical_vars)
    fe_dummy = pd.get_dummies(X[i])
    cate_list.append(fe_dummy)
    
dummy_all = pd.concat(cate_list, axis = 1)
dummy_all.head()
{% endcode %}