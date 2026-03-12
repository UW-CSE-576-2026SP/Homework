# Welcome to CSE 576, 2026 Spring

In this repository, you will find instructions on how to build your own image processing/computer vision library from (mostly) scratch. The work is divided out into different homework assignments, found in the `src/` directory.

To get started, make sure you have `git` and Python installed. Then run:

```
git clone https://github.com/UW-CSE-576-2024SP/Homework.git
cd Homework
```

and check to see that everything runs correctly. We recommend using Linux or MacOS for the homework for a smoother setup.

## Due Dates
** HW1 is due on April 8 (11:59 pm).**

** HW2 is due on April 17 (11:59 pm).**

** HW3 is due on May 1 (11:59 pm).**

** HW4 is due on May 8 (11:59 pm).**

** HW5 is due on May 20 (11:59 pm).**

## Get started on HW1

Open up the README for homework 1 in src/hw1/README.md, or view it [here](src/hw1/README.md). Good luck and have fun!

## Instructions for Mac Users
In MacOS, make sure Python 3.8+ is installed and available in your terminal.
If `python` is not found, install Python 3.8+ and run the tests with `python -m src.main test hw1`.

## Instructions for Windows Users
We do **NOT** recommend Windows OS for this assignment because environment setup can be more complex under Windows. However, if you only have Windows computers available, you can still manage your Python packages and environments with Anaconda.

## Setup

## Miniconda installation
Before you start working with this repo, you should install Anaconda.

**Before clicking in the link below** read notes below:

* Linux/Mac OS:
    * If using Linux/Mac please install command line version.
    * Make sure that you choose to initialize conda at startup.
        This will lead to fewer headaches in the future
* Windows:
    * If using Windows, we recommend using the Anaconda Terminal, which uses Bash-like syntax.
* Low storage system
    * If you are low of storage (<10GB; for example attu), then Miniconda (see link below) might be a better option.

### Download links

[Anaconda (default)](https://www.anaconda.com/products/individual#Downloads)

[Miniconda](https://docs.conda.io/en/latest/miniconda.html#latest-miniconda-installer-links) (if running low on disk space)

You can find more detailed instructions for installation [at this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html#installing-conda-on-a-system-that-has-other-python-installations-or-packages).

## Environment Setup
First make sure you have at least ~5GB of free drive.
Then, from this directory, run:
```
conda env create -f environment.yaml
conda activate cse576
```


