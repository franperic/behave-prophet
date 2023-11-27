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


@pytest.fixture(scope="module")
def get_prophet():
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
    return prophet


def test_all_methods(get_prophet):
    prophet = get_prophet
    prophet.evaluate()
    prophet.plot_summary()
    prophet.plot_interventions()
    prophet.plot_detail()
    prophet.plot_mapes()
