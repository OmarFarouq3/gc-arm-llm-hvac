"""Cleaning and derived metrics.

Paper: Section 3.3, step 1 (3-sigma test, frozen-sensor removal) and Eq. 1-5 (derived metrics such as COP).

Owner: Member A
"""


def load_raw(path):
    raise NotImplementedError


def remove_outliers(df, sigma=3):
    raise NotImplementedError


def remove_frozen_sensors(df, window):
    raise NotImplementedError


def add_derived_metrics(df):
    raise NotImplementedError
