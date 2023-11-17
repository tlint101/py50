# py50: Generate Dose-Response Curves

![Static Badge](https://img.shields.io/badge/py50%3A_Dose_Response-800000)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/py50)

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

## Installation
The package can be installed using the following:

```
pip install py50
```

Pacakge can be upgraded specifically using pip with the following:
```
pip install py50 -U
```

## Tutorial
A Jupyter Notebook demoing the code can be found [here](https://github.com/tlint101/py50/tree/main/tutorials).

A blog post demoing the code can be found at [Practice Coding]() (under development)

And finally, a documentation page set up through GitHub is currently in the works. 

## Citation
If you are interested in citing the file, I have generated a DOI link using Zenodo here:

## Future Work
I am interested in maintaining this for the foreseeable future. As such, I have several
things on my "To-Do" list. I will get around to them when I can. They are in no particular
order:

- Code for averaging and creating error bars on plot
- Code to generate EC50 plots
- Code to generate LD50 plots
- Housekeeping (I need to better understand Class composition!)

Thanks for your interest! 