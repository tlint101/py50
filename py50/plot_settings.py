"""
Color and Marker schemes for plotting
"""

# Color-blind safe palette from
# http://bconnelly.net/2013/10/creating-colorblind-friendly-figures
CBPALETTE = ('#000000', '#E69F00', '#56B4E9', '#009E73',
             '#F0E442', '#0072B2', '#D55E00', '#CC79A7')

# Matplotlib markers. Default is same length as CBPALETTE. More marker info can be found here:
# https://matplotlib.org/api/markers_api.html
CBMARKERS = ('o', '^', 's', 'D', 'v', '<', '>', 'p')
assert len(CBPALETTE) == len(CBMARKERS)

if __name__ == '__main__':
    import doctest

    doctest.testmod()
