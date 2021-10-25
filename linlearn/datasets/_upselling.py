import os
from os.path import exists, join
import logging

import numpy as np
import pandas as pd
import pickle
from sklearn.datasets._base import _fetch_remote


from .dataset import Dataset
from ._utils import _mkdirp, RemoteFileMetadata, get_data_home


ARCHIVE = RemoteFileMetadata(
    filename="KDDCup09_upselling.csv",
    url="https://www.openml.org/data/get_csv/53997/KDDCup09_upselling.arff",
    checksum="91115aa5a8b02ccee59feccb0dab3e45bcdaf023e838d2c54a5fb8f8940513b0",
)

logger = logging.getLogger(__name__)

def _fetch_upselling(download_if_missing=True):
    data_home = get_data_home()
    data_dir = join(data_home, "upselling")
    data_path = join(data_dir, "upselling.csv.gz")

    if download_if_missing and not exists(data_path):
        _mkdirp(data_dir)
        logger.info("Downloading %s" % ARCHIVE.url)
        _fetch_remote(ARCHIVE, dirname=data_dir)
        logger.debug("Converting as a single dataframe with the correct schema")
        filepath = join(data_dir, ARCHIVE.filename)

        df = pd.read_csv(filepath)
        def replace_categories(x):
            if x in [np.float64]:
                return np.float64
            elif x in [np.int64]:
                return np.int64
            else:
                return "category"
        dtype = {a: replace_categories(b) for a, b in zip(df.columns, df.dtypes)}
        dtype.pop("UPSELLING")

        with open(join(data_dir, "dtypes.pickle"), "wb") as f:
            pickle.dump(dtype, f)

        for col, typ in zip(df.columns, df.dtypes):
            if typ in [np.float64, np.int64]:
                df[col] = df[col].replace("?", np.nan)
            else:
                df[col] = df[col].astype("category")

        df.to_csv(data_path, compression="gzip", index=False)
        # Remove temporary files
        os.remove(filepath)


def load_upselling(download_if_missing=True):
    # Fetch the data is necessary
    _fetch_upselling(download_if_missing)

    data_home = get_data_home()
    data_dir = join(data_home, "upselling")
    data_path = join(data_dir, "upselling.csv.gz")

    with open(join(data_dir, "dtypes.pickle"), "rb") as f:
        dtypes = pickle.load(f)

    dataset = Dataset.from_dtype(
        name="upselling",
        drop_columns=None,
        task="binary-classification",
        label_column="UPSELLING",
        dtype=dtypes,
    )
    return dataset.load_from_csv(data_path, dtype=dtypes)