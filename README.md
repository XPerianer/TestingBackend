# TestingBackend
[![Build Status](https://travis-ci.com/XPerianer/TestingBackend.svg?branch=main)](https://travis-ci.com/XPerianer/TestingBackend)

This is the Backend for [TestingPlugin](https://github.com/XPerianer/TestingPlugin).
It provides query options for the relevance of tests and the option to run tests prioritized by predicted failures.


# Installation

## Requirements
* `Python>=3.8` is recommended
* Folder in which the Repository under test is located
* Virtual Environment that can run the tests for the repository using `pytest` with [pytest-immedate](https://github.com/XPerianer/pytest-immediate/) loaded

## Data sources
If you do not want to use the standard data folder that contains the files for the [flask repository](https://github.com/pallets/flask)
* Mutation Testing Dataset generated by [Mutester](https://github.com/XPerianer/Mutester) (.pkl)
* Predictor and test information generated by the [preprocessing notebook](https://github.com/XPerianer/ImmediateTestFeedback) (.joblib and .json)

## Dependencies
It's best to use a virtual environment. To create it and install all necessary dependencies, call
```
python -m venv venv
. venv/bin/activate
pip install -e . -r requirements.txt
```
inside the repository.

## Configuration
The file `config.cfg` decides from where TestingBackend loads its resources. Use absolute paths, and do not include `/` at the end for folders.

The two strings that control the loaded repository are:
* `REPOSITORY_PATH` The path to the repository that should be developed
* `VIRTUAL_ENVIRONMENT_PATH` The path to the according virtual environment with `pytest` and `pytest-immediate` loaded

If you do not use the standard provided files, also update these pathes:
* `PREDICTOR_MODEL_JOBLIB_FILE` The path to the `.joblib` file
* `MUTATION_TESTING_FILE` The path to the `.pkl` file
* `TEST_JSON_FILE` The path to the `.json` file

## Starting the server
```
python app.py
```
starts the server. At the beginning, some relevance information is prefetched from the dataset, which can take some seconds.
A simple check to test if the server is functional is to open [http://localhost:9001/data](http://localhost:9001/data) in a browser. This should return the `.json` set up in the configuration.

With the server up and running, everything is set up to open the [TestingPlugin](https://github.com/XPerianer/TestingPlugin)