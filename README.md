![py50_full.png](img/py50_full.png)

# py50: Generate Dose-Response Curves

![Static Badge](https://img.shields.io/badge/py50_v0.3.4-13406E)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/py50)
[![Documentation Status](https://readthedocs.org/projects/py50/badge/?version=latest)](https://py50.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/716929963.svg)](https://zenodo.org/doi/10.5281/zenodo.10183912)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://py50-app.streamlit.app)


## Summary
The project was created for laboratory use. I found many of my classmates/coworkers were 
using a program that I found to be unfriendly to generate dose-response curves. I found
a few other repositories that can also generate dose-response curves in python, however,
they did not meet my requirements:
1. Use Pandas for the Data so that it can be plugged into a Jupyter Notebook or Python scripts
2. Adaptable to user needs
3. Easy to use (hopefully!)


The curves are built on the four parameter logistic regression model:
$$Y = \text{Min} + \frac{\text{Max} - \text{Min}}{1 + \left(\frac{X}{\text{IC50}}\right)^{\text{Hill coefficient}}}$$
where min is the minimum response value, max is the maximum response value, Y is the response values of the curves, X 
is the concentration.  


This project meets my and my lab mates' needs. Hopefully it can meet the needs of others.

## Installation


```
pip install py50
```

Pacakge can be upgraded specifically using pip with the following:
```
pip install py50 -U
```

## Tutorial
Documentation can be found [here](https://py50.readthedocs.io/en/latest/).

A Jupyter Notebook demoing the code can be found [here](https://github.com/tlint101/py50/tree/main/tutorials).

A blog post demoing the code can be found at [Practice in Code](https://tlint101.github.io/practice-in-code/)

# Web Application [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://py50-app.streamlit.app)
For those who are not versed in python coding, py50 has been converted into a web application using Streamlit!
The web application can be found here: [py50-app](https://py50-app.streamlit.app)

## Citation
If you are interested in citing the file, I have generated a DOI link using Zenodo here: [![DOI](https://zenodo.org/badge/716929963.svg)](https://zenodo.org/doi/10.5281/zenodo.10183912)

## Future Work
I am interested in maintaining this for the foreseeable future. As such, I have several
things on my "To-Do" list. I will get around to them when I can. In no particular
order:

- [ ] Code for averaging and creating error bars on plot
- [ ] Include scripts for statistics 
- [ ] Include scripts for plotting and annotating with values from statistics calculations
- [ ] Update statistic calculations for Streamlit app
- [ ] Update plot functions for Streamlit app
- [X] Refactor code for maintainability (I think I understand classes!)

Thanks for your interest! 