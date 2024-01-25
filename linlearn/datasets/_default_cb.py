import os
from os.path import exists, join
import logging

import numpy as np
import pandas as pd
from sklearn.datasets._base import _fetch_remote

from .dataset import Dataset
from ._utils import _mkdirp, RemoteFileMetadata, get_data_home


ARCHIVE = RemoteFileMetadata(
    filename="default_cb.xls",
    url="https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls",
    checksum="30c6be3abd8dcfd3e6096c828bad8c2f011238620f5369220bd60cfc82700933",
)

logger = logging.getLogger(__name__)


def _fetch_default_cb(download_if_missing=True):
    data_home = get_data_home()
    data_dir = join(data_home, "default_cb")
    data_path = join(data_dir, "default_cb.csv.gz")

    if download_if_missing and not exists(data_path):
        _mkdirp(data_dir)
        logger.info("Downloading %s" % ARCHIVE.url)
        _fetch_remote(ARCHIVE, dirname=data_dir)
        logger.debug("Converting as a single dataframe with the correct schema")
        filepath = join(data_dir, ARCHIVE.filename)

        df = pd.read_excel(filepath, skiprows=[0])

        df.to_csv(data_path, compression="gzip", index=False)
        # Remove temporary files
        os.remove(filepath)


def load_default_cb(download_if_missing=True):
    # Fetch the data is necessary
    _fetch_default_cb(download_if_missing)

    data_home = get_data_home()
    data_dir = join(data_home, "default_cb")
    data_path = join(data_dir, "default_cb.csv.gz")

    dtype = {
        "LIMIT_BAL": int,
        "SEX": "category",
        "EDUCATION": "category",
        "MARRIAGE": "category",
        "AGE": int,
        # We consider the PAY_* features as continuous, otherwise some modalities are
        # very rare and lead to problems in train/test splitting
        "PAY_0": int,
        "PAY_2": int,
        "PAY_3": int,
        "PAY_4": int,
        "PAY_5": int,
        "PAY_6": int,
        "BILL_AMT1": int,
        "BILL_AMT2": int,
        "BILL_AMT3": int,
        "BILL_AMT4": int,
        "BILL_AMT5": int,
        "BILL_AMT6": int,
        "PAY_AMT1": int,
        "PAY_AMT2": int,
        "PAY_AMT3": int,
        "PAY_AMT4": int,
        "PAY_AMT5": int,
        "PAY_AMT6": int,
    }
    dataset = Dataset.from_dtype(
        name="default-cb",
        task="binary-classification",
        label_column="default payment next month",
        dtype=dtype,
        drop_columns=["ID"],
    )
    return dataset.load_from_csv(data_path, dtype=dtype)
