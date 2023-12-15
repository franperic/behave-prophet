# Behave-prophet (Behavior testing for Prophet)

![logo](logo.png)

# Prophet Behavior Testing

[![codecov](https://codecov.io/gh/franperic/flair/graph/badge.svg?token=6P7C5HHPOC)](https://codecov.io/gh/franperic/flair)

# Getting started

## Context

Check out this [blogpost](https://franperic.github.io/posts/flair/) for an introduction to behavior testing in the context of time series foreacsting.

## Prerequisites

This library is using [Prophet](https://facebook.github.io/prophet) as the forecasting engine and [Postgres](https://www.postgresql.org/) as the database.

In order to use the functionality of this library you will need Postgres installed and create a table with the schema described in the [schema.sql](schema.sql) file.

## Installation

You can install this library using pip:

```bash
pip install behave-prophet
```

```python
from behave_prophet.prophet_evaluator import ProphetEvaluator
```

## Examples

In the subfolder `examples` you can find a simple example of how to use this library.

### Credits

This project was inspired by [this blogpost](https://www.jeremyjordan.me/testing-ml/) about testing machine learning models.
