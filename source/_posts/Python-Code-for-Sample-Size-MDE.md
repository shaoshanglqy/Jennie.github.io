---
title: Python for Sample Size & MDE Calculation
date: 2019-12-09 20:19:04
tags: ['Experiment', 'Statistic', 'Python']
categories: ['Experiment']
---
<font size=3>

### Sample Size Calculation

{% code lang:python %}
import scipy.stats as stats

def sample_size_calculation(mu, sigma, MDE, alpha=0.05, beta=0.2):
    return 2 * (sigma**2) * ((stats.norm.ppf(1-alpha/2) + stats.norm.ppf(1-beta))**2) / ((mu * MDE)**2)

{% endcode %}


### Minimum Defective Effect
{% code lang:python %}
from scipy.stats import norm

sample_size = 1000
alpha = 0.05
z = norm.isf(alpha / 2)
estimated_variance = ds.y.var()
detectable_effect_size = z * np.sqrt(2 * estimated_variance / sample_size)
{% endcode %}

</font>