MCP Server (AI Clients)
========================

``py50`` ships a local Model Context Protocol (MCP) server (``py50-mcp``) over
stdio. It lets AI assistants inspect ``.csv``, ``.tsv``, and ``.xlsx`` assay
tables, compute 4-parameter logistic (4PL) IC50 / pIC50 values, run statistical
tests, and return PNG plots alongside numeric summaries.

Installation
------------

**Zero-install launch with uv (recommended):**

.. code-block:: bash

   uvx --from 'py50[mcp]' py50-mcp

**Persistent environment install with pip:**

.. code-block:: bash

   pip install 'py50[mcp]'

**Testing directly from a local git checkout (before PyPI release):**

.. code-block:: bash

   uv --directory /absolute/path/to/py50 run py50-mcp

**Optional file-output restriction:**

By default, plot tools return PNG images inline and only write a 300 dpi file
when ``savepath`` is passed. Set ``PY50_MCP_OUTPUT_DIR`` in the server
environment to restrict ``savepath`` writes to a single directory:

.. code-block:: bash

   PY50_MCP_OUTPUT_DIR=/absolute/path/to/outputs

Available Commands (Tools)
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 15 23 34 28

   * - Command (Tool)
     - What it does
     - Calculations / Tables Returned
     - Images Returned
   * - ``read_csv``
     - Inspects ``.csv``, ``.tsv``, or ``.xlsx`` files
     - Column names, dtypes, null counts, compound/group candidates, Excel worksheet list (``sheets``), and row preview
     - None
   * - ``calculate_ic50``
     - Fits 4PL curves (single or replicate columns) with unit conversion
     - ``maximum``, ``minimum``, ``relative ic50``, ``absolute ic50`` (``null`` if curve never crosses 50%), ``hill_slope``, ``relative pIC50``, ``absolute pIC50``
     - None
   * - ``single_curve``
     - Plots one compound
     - Full 4PL IC50 / pIC50 / Hill table for the compound
     - Inline PNG (scatter or replicate ``sd``/``se``/``ci``/``pi`` error bars, auto-scaled x-range, optional IC50 guide lines) + optional 300 dpi PNG
   * - ``multi_curve``
     - Overlays all compounds on shared axes
     - Full 4PL IC50 / pIC50 / Hill table for all compounds
     - Inline PNG (colorblind-safe palette & markers, optional IC50 box on ``box_target``) + optional 300 dpi PNG
   * - ``grid_curve``
     - Plots one subplot per compound in a grid
     - Full 4PL IC50 / pIC50 / Hill table for all compounds
     - Inline PNG subplot grid (1--6 columns, auto-averaged replicates, optional IC50 boxes) + optional 300 dpi PNG
   * - ``run_stats``
     - Runs any of 16 statistical tests
     - Markdown + JSON table for the chosen test (see supported tests below)
     - None
   * - ``stats_plot``
     - Draws any of 11 statistical plot types
     - Underlying post-hoc / Tukey table or p-value matrix (all plots except histograms and QQ plots)
     - Inline PNG + optional 300 dpi PNG

Supported ``run_stats`` Tests (16)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Assumptions:** ``normality``, ``homoscedasticity``
* **Omnibus:** ``anova``, ``welch-anova``, ``rm-anova``, ``mixed-anova``, ``kruskal``, ``friedman``
* **Post-hoc & Pairwise:** ``tukey``, ``gameshowell``, ``wilcoxon``, ``mannu``, ``pairwise-parametric``, ``pairwise-nonparametric``, ``pairwise-rm``, ``pairwise-mixed`` (with ``padjust``: ``none``, ``bonf``, ``sidak``, ``holm``, ``fdr_bh``, ``fdr_by``)

Supported ``stats_plot`` Plot Types (11)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Categorical with significance brackets:** ``boxplot``, ``barplot``, ``violinplot``, ``swarmplot``, ``stripplot``, ``boxenplot``, ``pointplot`` (supports ``tukey``, ``gameshowell``, ``wilcoxon``, ``mannu``, ``pairwise-parametric``, ``pairwise-nonparametric``, ``pairwise-rm``, ``pairwise-mixed``)
* **Confidence intervals & matrices:** ``ci_plot`` (Tukey HSD CI + ANOVA p-value), ``p_matrix`` (p-value heatmap)
* **Distributions:** ``histplot``, ``qqplot``

Client Setup
------------

Antigravity
~~~~~~~~~~~

Add ``py50`` to ``~/.gemini/config/mcp_config.json``:

.. code-block:: json

   {
     "mcpServers": {
       "py50": {
         "command": "uvx",
         "args": ["--from", "py50[mcp]", "py50-mcp"]
       }
     }
   }

Claude Desktop & Claude Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For **Claude Desktop**, add ``py50`` to ``claude_desktop_config.json``:

.. code-block:: json

   {
     "mcpServers": {
       "py50": {
         "command": "uvx",
         "args": ["--from", "py50[mcp]", "py50-mcp"]
       }
     }
   }

For **Claude Code**, register via CLI or add the same JSON block to ``.mcp.json``
in your project root:

.. code-block:: bash

   claude mcp add py50 -- uvx --from 'py50[mcp]' py50-mcp

Codex
~~~~~

Add ``py50`` to ``~/.codex/config.toml`` (global) or ``.codex/config.toml``
(project):

.. code-block:: toml

   [mcp_servers.py50]
   command = "uvx"
   args = ["--from", "py50[mcp]", "py50-mcp"]

OpenCode 2
~~~~~~~~~~

Add ``py50`` under ``"mcp"`` in ``~/.config/opencode/opencode.json`` (global) or
``./opencode.json`` (project):

.. code-block:: json

   {
     "$schema": "https://opencode.ai/config.json",
     "mcp": {
       "py50": {
         "type": "local",
         "command": ["uvx", "--from", "py50[mcp]", "py50-mcp"],
         "enabled": true
       }
     }
   }

Jan & LM Studio
~~~~~~~~~~~~~~~

In Jan or LM Studio's MCP configuration, add the standard ``mcpServers`` block:

.. code-block:: json

   {
     "mcpServers": {
       "py50": {
         "command": "uvx",
         "args": ["--from", "py50[mcp]", "py50-mcp"]
       }
     }
   }

Notes
-----

* **Absolute paths:** The MCP server has no working directory, so always pass
  absolute paths for ``filepath`` and optional ``savepath``.
* **Concentration units:** ``input_units`` declares the unit of the input table
  (``pM``, ``nM``, or ``µM`` / ``uM``; default ``nM``). Optional ``output_units``
  converts reported IC50 values and the plot x-axis without altering pIC50.
* **Excel worksheets:** Pass ``sheet`` when reading a specific tab of an
  ``.xlsx`` workbook; if omitted, the first worksheet is used.
