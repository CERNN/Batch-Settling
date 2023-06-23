# Batch-Settling

Module for simulating particles settling in non-Newtonian fluid. It builds the source code from a python code to C, using cython for performance fixes.

## Setup and installation

To setup your environment to run the scripts, it's required Python and Poetry.

Poetry can be installed by running

```bash
 pip install poetry
```

After that, to install the project dependencies

```bash
 poetry install
```

And then run scripts using

```bash
 poetry run <command-you-want-to-run>
 # Such as
 poetry run python cythonExtension.py build_ext --inplace # To build cython code
 poetry run python MVF\PowerLawFluid.py # To run the simulation
 ...
```