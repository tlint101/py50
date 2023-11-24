# py50

py50 is used to generate dose-response curves. The package can be installed using the following:

```
pip install py50
```

## Summary
The project was created to laboratory use. I found many of my classmates/coworkers were 
using a program that I found to be unfriendly to generate dose-response curves. I found
a few other repositories that can also generate dose-response curves in python, however,
they did not meet my requirements:
1. Use Pandas for the Data so that it can be plugged into a typical Jupyter Notebook or Python scripts
2. Adaptable to user needs
3. Easy to use (hopefully!)

The curves are built on the four parameter logistic regression model:
$$Y = \text{Min} + \frac{\text{Max} - \text{Min}}{1 + \left(\frac{X}{\text{IC50}}\right)^{\text{Hill coefficient}}}$$
where min is the minimum response value, max is the maximum response value, Y is the response values of the curves, X 
is the concentration.  


This project meets our needs. And hopefully it can meet the needs of others.

Additional documentation can be found here. The repository and the tutorials can be found at the following:
https://github.com/tlint101/py50