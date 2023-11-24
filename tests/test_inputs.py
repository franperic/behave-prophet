import os
import sys

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the parent directory to Python path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import pytest
from flair.prophet_evaluator import ProphetEvaluator
import pandas as pd


def test_valid_inputs():
    prophet = ProphetEvaluator(
        df=pd.DataFrame(
            {
                "date": pd.date_range("2005-01-01", periods=1000, freq="M"),
                "sales": np.random.normal(size=1000),
            }
        ),
        target="sales",
        date="date",
        testtype="structural",
    )
    prophet.evaluate()
