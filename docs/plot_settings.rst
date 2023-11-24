.. plot_settings:

Plot Settings
=====================

Default color palette and marker settings.

.. automodule:: plot_settings
   :members:
   :undoc-members:

.. data:: CBPALETTE

   Color-blind safe palette from http://bconnelly.net/2013/10/creating-colorblind-friendly-figures.

   Type: tuple

   .. rubric:: Default value:

   ('#000000', '#E69F00', '#56B4E9', '#009E73',
    '#F0E442', '#0072B2', '#D55E00', '#CC79A7')

.. data:: CBMARKERS

   Matplotlib markers. Default is the same length as CBPALETTE. More marker info can be found
   `here <https://matplotlib.org/api/markers_api.html>`_.

   Type: tuple

   .. rubric:: Default value:

   ('o', '^', 's', 'D', 'v', '<', '>', 'p')
