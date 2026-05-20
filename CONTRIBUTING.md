# Contributing to CheMLFlow

Thanks for your interest in contributing to **CheMLFlow**!

We welcome contributions of all kinds — bug fixes, new features, documentation improvements, tests, and more.

---

## How to contribute

Follow the steps below:

### 1. Fork the repository

Go to [https://github.com/nijamudheen/CheMLFlow.git](https://github.com/nijamudheen/CheMLFlow.git)  
and click **Fork** (top-right corner).

This creates your own copy of the repo under your GitHub account.

---

### 2. Clone your fork locally

git clone https://github.com/nijamudheen/CheMLFlow.git

cd yourforkname

### 3. Create a new branch for your work

git checkout -b feature/my-feature

Use a descriptive branch name (e.g., bugfix/fix-typo, feature/add-plotting).

### 4. Set up the development environment

We recommend using Conda for dependencies:

conda create -n myenv python=3.13

conda activate myenv

pip install -e .[dev]

conda install -c conda-forge rdkit

This installs the package in editable mode and developer dependencies.

### 5. Make your changes

Edit code, add tests, improve docs — anything you like!


### 6. Run tests

See tests directory (currently needs own data or get data from ChemBL using scripts in GetData folder! Automated tests options will be released soon!)

### 7. Commit and push your changes

git add .

git commit -m "Your clear, descriptive commit message"

git push origin feature/my-feature


### 8. Open a Pull Request (PR)

Go to your fork on GitHub → click "Compare & pull request"

Describe your changes → submit!

We will review it and work with you to merge.

## Guidelines

Follow existing code style and formatting

Write clear commit messages

Add/maintain tests where applicable

Keep PRs focused and limited in scope

## Reporting issues

Found a bug or have a feature request?

Please open an issue describing:

What happened

Steps to reproduce

Expected behavior

Environment details

## Thank you!

Every contribution makes this project better — thank you for your time and effort!












