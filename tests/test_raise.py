import os
import sys

# Get the current directory of the script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Append the parent directory to Python path
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import pytest
from behave_prophet.prophet_evaluator import ProphetEvaluator
import pandas as pd


def test_arguments():
    with pytest.raises(TypeError):
        ProphetEvaluator(
            df=pd.DataFrame(
                {
                    "ds": pd.date_range("2005-01-01", preiods=1000, freq="M"),
                    "y": np.random.normal(1000),
                }
            ),
        )


def test_static_ts():
    with pytest.raises(Exception):
        ProphetEvaluator(
            df=pd.DataFrame(
                {
                    "ds": pd.date_range("2005-01-01", preiods=1000, freq="M"),
                    "y": 54,
                }
            ),
            target="y",
            date="ds",
            testtype="structural",
        )


def test_short_ts():
    with pytest.raises(TypeError):
        ProphetEvaluator(
            df=pd.DataFrame(
                {
                    "ds": pd.date_range("2005-01-01", preiods=12, freq="M"),
                    "y": np.random.normal(12),
                }
            ),
            target="y",
            date="ds",
            testtype="structural",
        )
