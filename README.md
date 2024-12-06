![py50_full.png](img/py50_full.png)

# py50: Generate Dose-Response Curves

![Static Badge](https://img.shields.io/badge/py50_v1.0.9-13406E)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.34.0-FF4B4B.svg?style=flat&logo=Streamlit&logoColor=white)](https://py50-app.streamlit.app)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/py50?style=flat&logo=python&logoColor=white)
[![Documentation Status](https://readthedocs.org/projects/py50/badge/?version=latest)](https://py50.readthedocs.io/en/latest/?badge=latest)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![DOI](https://zenodo.org/badge/716929963.svg)](https://zenodo.org/doi/10.5281/zenodo.10183912)

## Summary

The aim of py50 is to make the generation of dose-respnose curves and annotated plots with statistics. The project was
created primarily for my personal use and for my coworkers/classmates. I found many of my classmates/coworkers were
using a program that I find to be unfriendly in generating dose-response curves or with calculating statistics and
plots. During my search, I found other helpful repositories that can generate dose-response curves, calculate
statistics, or make annotated plots. However, I found that these packages did not meet my requirements:

1. Use Pandas for the Data so that it can be easily plugged into a Jupyter Notebook or Python scripts
2. Adaptable to user needs
3. Easy to use (hopefully!)

The dose-response curves in py50 are built using the four parameter logistic regression model:
$$Y = \text{Min} + \frac{\text{Max} - \text{Min}}{1 + \left(\frac{X}{\text{IC50}}\right)^{\text{Hill coefficient}}}$$
where min is the minimum response value, max is the maximum response value, Y is the response values of the curves, and 
X is the concentration.

The statistics and annotated plots are wrapped from [Pingouin](https://github.com/raphaelvallat/pingouin)
and [Statannotations](https://github.com/trevismd/statannotations).
This may have been done inelegantly and will be updated based on my use or recommendations by other. As things stand, 
this project meets my and the needs of my classmates/coworkers. Hopefully it can meet the needs of others.

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

The repository for the Streamlit app version can be found
here: [py50-streamlit](https://github.com/tlint101/py50-streamlit)

**NOTE:** Updates to the web application take more time. Updates will be made when possible or upon request.

## Future Work

With the release of py50 v1.0.0, I have finished a project that has been on my mind for the past six months. My aim now
will be to reformat the code for maintainability and to fix any bugs that I find or others report. I plan on maintaining
py50 for the foreseeable future. As such, my current "To-Do" list (in no particular order) are as follows:

- [ ] Complete To-Do notes in Python script
- [X] Update Tutorials for clarity
- [X] Update py50 Streamlit to version 1.0.0
- [ ] Refactor code for maintainability
- [ ] **Add error messages!**
- [ ] (Bonus Points) Provide KNIME workflow?

## Citation

If you are interested in citing the repository, I have generated a DOI link using Zenodo
here: [![DOI](https://zenodo.org/badge/716929963.svg)](https://zenodo.org/doi/10.5281/zenodo.10183912)

For those using a citation manager, there is also a compressed endnote file that is available and can be
downloaded [here](https://github.com/tlint101/py50/tree/v1.0.4/citation)

Thanks for your interest! 