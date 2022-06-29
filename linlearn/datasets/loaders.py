# Authors: Stephane Gaiffas <stephane.gaiffas@gmail.com>
# License: BSD 3 clause

import os

import pandas as pd
import numpy as np

from .dataset import Dataset
from ._adult import load_adult
from ._bank import load_bank
from ._car import load_car
from ._default_cb import load_default_cb

# TODO: kdd98 https://www.openml.org/d/23513, Il y a plein de features numeriques avec un grand nombre de "missing" values


def load_breastcancer():
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)
    df = data["frame"]

    continuous_columns = [col for col in df.columns if col != "target"]
    categorical_columns = None
    dataset = Dataset(
        name="breastcancer",
        task="binary-classification",
        label_column="target",
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
    )
    dataset.df_raw = df
    return dataset


def load_boston():
    from sklearn.datasets import load_boston

    data = load_boston()
    # Load as a dataframe. We set some features as categorical so that we have a
    # regression example with categorical features for tests...
    df = pd.DataFrame(data["data"], columns=data["feature_names"]).astype(
        {"CHAS": "category"}
    )
    df["target"] = data["target"]
    continuous_columns = [
        "ZN",
        "RAD",
        "CRIM",
        "INDUS",
        "NOX",
        "RM",
        "AGE",
        "DIS",
        "TAX",
        "PTRATIO",
        "B",
        "LSTAT",
    ]
    categorical_columns = ["CHAS"]
    dataset = Dataset(
        name="boston",
        task="regression",
        label_column="target",
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
    )
    dataset.df_raw = df
    return dataset


def load_californiahousing():
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    df = data["frame"]
    continuous_columns = [col for col in df.columns if col != "MedHouseVal"]
    categorical_columns = []
    dataset = Dataset(
        name="californiahousing",
        task="regression",
        label_column="MedHouseVal",
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
    )
    dataset.df_raw = df
    return dataset


def load_letor():
    dtype = {str(i): np.float for i in range(1, 47)}
    dataset = Dataset.from_dtype(
        name="letor",
        task="multiclass-classification",
        label_column="0",
        dtype=dtype,
    )
    return dataset.load_from_csv("letor.csv.gz", dtype=dtype)


def load_cardio():
    dtype = {
        "b": np.int,
        "e": np.int,
        "LBE": np.int,
        "LB": np.int,
        "AC": np.int,
        "FM": np.int,
        "UC": np.int,
        "ASTV": np.int,
        "MSTV": np.float,
        "ALTV": np.int,
        "MLTV": np.float,
        "DL": np.int,
        "DS": np.int,
        "DP": np.int,
        "DR": np.int,
        "Width": np.int,
        "Min": np.int,
        "Max": np.int,
        "Nmax": np.int,
        "Nzeros": np.int,
        "Mode": np.int,
        "Mean": np.int,
        "Median": np.int,
        "Variance": np.int,
        "Tendency": np.int,
        "A": np.int,
        "B": np.int,
        "C": np.int,
        "D": np.int,
        "E": np.int,
        "AD": np.int,
        "DE": np.int,
        "LD": np.int,
        "FS": np.int,
        "SUSP": np.int,
    }
    dataset = Dataset.from_dtype(
        name="cardio",
        task="multiclass-classification",
        label_column="CLASS",
        dtype=dtype,
        # We drop the NSP column which is a 3-class version of the label
        drop_columns=["FileName", "Date", "SegFile", "NSP"],
    )
    return dataset.load_from_csv(
        "cardiotocography.csv.gz", sep=";", decimal=",", dtype=dtype
    )


def load_churn():
    dtype = {
        "State": "category",
        "Account Length": np.int,
        "Area Code": "category",
        "Int'l Plan": "category",
        "VMail Plan": "category",
        "VMail Message": np.int,
        "Day Mins": np.float,
        "Day Calls": np.int,
        "Day Charge": np.float,
        "Eve Mins": np.float,
        "Eve Calls": np.int,
        "Eve Charge": np.float,
        "Night Mins": np.float,
        "Night Calls": np.int,
        "Night Charge": np.float,
        "Intl Mins": np.float,
        "Intl Calls": np.int,
        "Intl Charge": np.float,
        "CustServ Calls": np.int,
    }
    dataset = Dataset.from_dtype(
        name="churn",
        task="binary-classification",
        label_column="Churn?",
        dtype=dtype,
        # We drop the "Phone" column
        drop_columns=["Phone"],
    )
    return dataset.load_from_csv("churn.csv.gz", dtype=dtype)


def load_electrical():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/Electrical+Grid+Stability+Simulated+Data+#
    dtype = {
        "tau1": np.float,
        "tau2": np.float,
        "tau3": np.float,
        "tau4": np.float,
        "p1": np.float,
        "p2": np.float,
        "p3": np.float,
        "p4": np.float,
        "g1": np.float,
        "g2": np.float,
        "g3": np.float,
        "g4": np.float,
        "stab": np.float,
    }
    dataset = Dataset.from_dtype(
        name="electrical",
        task="binary-classification",
        label_column="stabf",
        dtype=dtype,
    )
    return dataset.load_from_csv("electrical.csv.gz", dtype=dtype)


def load_occupancy():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+
    # the dataframes obtained from files datatraining.txt, datatest.txt and datatest2.txt were
    # concatenated in this order with ignore_index = True
    dtype = {
        "Temperature": np.float,
        "Humidity": np.float,
        "Light": np.float,
        "CO2": np.float,
        "HumidityRatio": np.float,
    }
    dataset = Dataset.from_dtype(
        name="occupancy",
        task="binary-classification",
        label_column="Occupancy",
        dtype=dtype,
        drop_columns=["date"],
    )
    return dataset.load_from_csv("occupancy.csv.gz", dtype=dtype)


def load_avila():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/Avila
    # the dataframes obtained from files avila-tr.txt and avila-ts.txt were
    # concatenated in this order with ignore_index = True
    dtype = {
        "0": np.float,
        "1": np.float,
        "2": np.float,
        "3": np.float,
        "4": np.float,
        "5": np.float,
        "6": np.float,
        "7": np.float,
        "8": np.float,
        "9": np.float,
    }
    dataset = Dataset.from_dtype(
        name="avila",
        task="multiclass-classification",
        label_column="10",
        dtype=dtype,
    )
    return dataset.load_from_csv("avila.csv.gz", dtype=dtype)


def load_miniboone():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/MiniBooNE+particle+identification
    # parsed with miniboone_parser.py
    dtype = {str(i): np.float for i in range(50)}
    dataset = Dataset.from_dtype(
        name="miniboone",
        task="binary-classification",
        label_column="50",
        dtype=dtype,
    )
    return dataset.load_from_csv("miniboone.csv.gz", dtype=dtype)


def load_eeg():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/EEG+Eye+State
    # parsed with pd.read_csv(..., skiprows=19, header=None)
    dtype = {str(i): np.float for i in range(14)}
    dataset = Dataset.from_dtype(
        name="eeg",
        task="binary-classification",
        label_column="14",
        dtype=dtype,
    )
    return dataset.load_from_csv("eeg_eye.csv.gz", dtype=dtype)


def load_drybean():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/Dry+Bean+Dataset
    # parsed with pd.read_csv(..., skiprows=25, header=None)
    dtype = {str(i): np.float for i in range(16)}
    dataset = Dataset.from_dtype(
        name="drybean",
        task="multiclass-classification",
        label_column="16",
        dtype=dtype,
    )
    return dataset.load_from_csv("drybean.csv.gz", dtype=dtype)


def load_gas():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/Gas+Sensor+Array+Drift+Dataset
    # parsed with gas_parser.py
    dtype = {str(i): np.float for i in range(1, 129)}
    dataset = Dataset.from_dtype(
        name="gas",
        task="multiclass-classification",
        label_column="0",
        dtype=dtype,
    )
    return dataset.load_from_csv("gas.csv.gz", dtype=dtype)


def load_energy():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/Appliances+energy+prediction
    # directly extracted with pd.read_csv and compressed into gzip format
    dtype = {
        # "date",
        # "Appliances",
        "lights": np.float,
        "T1": np.float,
        "RH_1": np.float,
        "T2": np.float,
        "RH_2": np.float,
        "T3": np.float,
        "RH_3": np.float,
        "T4": np.float,
        "RH_4": np.float,
        "T5": np.float,
        "RH_5": np.float,
        "T6": np.float,
        "RH_6": np.float,
        "T7": np.float,
        "RH_7": np.float,
        "T8": np.float,
        "RH_8": np.float,
        "T9": np.float,
        "RH_9": np.float,
        "T_out": np.float,
        "Press_mm_hg": np.float,
        "RH_out": np.float,
        "Windspeed": np.float,
        "Visibility": np.float,
        "Tdewpoint": np.float,
        "rv1": np.float,
        "rv2": np.float,
    }
    dataset = Dataset.from_dtype(
        name="energy",
        task="regression",
        label_column="Appliances",
        dtype=dtype,
        drop_columns=["date"],
    )
    return dataset.load_from_csv("energy.csv.gz", dtype=dtype)


def load_bike():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset
    # directly extracted from hour.csv with pd.read_csv and compressed into gzip format
    dtype = {
        # "instant",
        # "dteday",
        "season": "category",
        # "yr",
        # "mnth",
        # "hr",
        "holiday": "category",
        "weekday": "category",
        "workingday": "category",
        "weathersit": "category",
        "temp": np.float,
        "atemp": np.float,
        "hum": np.float,
        "windspeed": np.float,
        # "casual",
        # "registered",
        # "cnt",
    }
    dataset = Dataset.from_dtype(
        name="bike",
        task="regression",
        label_column="cnt",
        dtype=dtype,
        drop_columns=["instant", "dteday", "yr", "mnth", "hr", "casual", "registered"],
        # we keep only season and drop month, we discard the hour to avoid too many categories
    )
    return dataset.load_from_csv("bike.csv.gz", dtype=dtype)


def load_cbm():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/Condition+Based+Maintenance+of+Naval+Propulsion+Plants
    # parsed with cbm_parser.py and saved with pandas.to_csv(..., index=False, compression="gzip")
    dtype = {str(i): np.float for i in range(16)}
    dataset = Dataset.from_dtype(
        name="cbm",
        task="regression",
        label_column="16",
        dtype=dtype,
        drop_columns=["17"],
    )
    return dataset.load_from_csv("cbm.csv.gz", dtype=dtype)


def load_ccpp():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/Combined+Cycle+Power+Plant
    # converted to csv in Numbers then parsed with ccpp_parser.py
    dtype = {
        "AT": np.float,
        "V": np.float,
        "AP": np.float,
        "RH": np.float,
    }
    dataset = Dataset.from_dtype(
        name="ccpp",
        task="regression",
        label_column="PE",
        dtype=dtype,
    )
    return dataset.load_from_csv("ccpp.csv.gz", dtype=dtype)


def load_gasturbine():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/Gas+Turbine+CO+and+NOx+Emission+Data+Set
    # parsed with gasturbine_parser.py
    dtype = {
        "AT": np.float,
        "AP": np.float,
        "AH": np.float,
        "AFDP": np.float,
        "GTEP": np.float,
        "TIT": np.float,
        "TAT": np.float,
        "CDP": np.float,
        "CO": np.float,
        "NOX": np.float,
    }
    dataset = Dataset.from_dtype(
        name="gasturbine",
        task="regression",
        label_column="TEY",
        dtype=dtype,
    )
    return dataset.load_from_csv("gasturbine.csv.gz", dtype=dtype)


def load_metro():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/Metro+Interstate+Traffic+Volume
    # date_time column has been preprocessed into (hour - 12)**2
    dtype = {'temp': np.float, 'rain_1h': np.float, 'snow_1h': np.float, 'clouds_all': np.int, 'weather_main': "category", 'date_time': np.float}
    dataset = Dataset.from_dtype(
        name="metro",
        task="regression",
        label_column='traffic_volume',
        dtype=dtype,
        drop_columns=['holiday', 'weather_description']
    )
    return dataset.load_from_csv("metro.csv.gz", dtype=dtype)

def load_casp():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/Physicochemical+Properties+of+Protein+Tertiary+Structure#
    # directly read with pandas and compressed
    dtype = {'F%d' % i: np.float for i in range(1, 10)}
    dataset = Dataset.from_dtype(
        name="casp",
        task="regression",
        label_column='RMSD',
        dtype=dtype,
    )
    return dataset.load_from_csv("casp.csv.gz", dtype=dtype)

def load_superconduct():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/Superconductivty+Data
    # directly read with pandas and compressed
    feature_list = ['number_of_elements', 'mean_atomic_mass', 'wtd_mean_atomic_mass', 'gmean_atomic_mass', 'wtd_gmean_atomic_mass', 'entropy_atomic_mass', 'wtd_entropy_atomic_mass', 'range_atomic_mass', 'wtd_range_atomic_mass', 'std_atomic_mass', 'wtd_std_atomic_mass', 'mean_fie', 'wtd_mean_fie', 'gmean_fie', 'wtd_gmean_fie', 'entropy_fie', 'wtd_entropy_fie', 'range_fie', 'wtd_range_fie', 'std_fie', 'wtd_std_fie', 'mean_atomic_radius', 'wtd_mean_atomic_radius', 'gmean_atomic_radius', 'wtd_gmean_atomic_radius', 'entropy_atomic_radius', 'wtd_entropy_atomic_radius', 'range_atomic_radius', 'wtd_range_atomic_radius', 'std_atomic_radius', 'wtd_std_atomic_radius', 'mean_Density', 'wtd_mean_Density', 'gmean_Density', 'wtd_gmean_Density', 'entropy_Density', 'wtd_entropy_Density', 'range_Density', 'wtd_range_Density', 'std_Density', 'wtd_std_Density', 'mean_ElectronAffinity', 'wtd_mean_ElectronAffinity', 'gmean_ElectronAffinity', 'wtd_gmean_ElectronAffinity', 'entropy_ElectronAffinity', 'wtd_entropy_ElectronAffinity', 'range_ElectronAffinity', 'wtd_range_ElectronAffinity', 'std_ElectronAffinity', 'wtd_std_ElectronAffinity', 'mean_FusionHeat', 'wtd_mean_FusionHeat', 'gmean_FusionHeat', 'wtd_gmean_FusionHeat', 'entropy_FusionHeat', 'wtd_entropy_FusionHeat', 'range_FusionHeat', 'wtd_range_FusionHeat', 'std_FusionHeat', 'wtd_std_FusionHeat', 'mean_ThermalConductivity', 'wtd_mean_ThermalConductivity', 'gmean_ThermalConductivity', 'wtd_gmean_ThermalConductivity', 'entropy_ThermalConductivity', 'wtd_entropy_ThermalConductivity', 'range_ThermalConductivity', 'wtd_range_ThermalConductivity', 'std_ThermalConductivity', 'wtd_std_ThermalConductivity', 'mean_Valence', 'wtd_mean_Valence', 'gmean_Valence', 'wtd_gmean_Valence', 'entropy_Valence', 'wtd_entropy_Valence', 'range_Valence', 'wtd_range_Valence', 'std_Valence', 'wtd_std_Valence']#, 'critical_temp']
    dtype = {x: np.float for x in feature_list}
    dataset = Dataset.from_dtype(
        name="superconduct",
        task="regression",
        label_column='critical_temp',
        dtype=dtype,
    )
    return dataset.load_from_csv("superconduct.csv.gz", dtype=dtype)

def load_sgemm():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/SGEMM+GPU+kernel+performance
    # read with pandas and added column runs_avg as average of Run 1 2 3 4

    feature_list = ['MWG', 'NWG', 'KWG', 'MDIMC', 'NDIMC', 'MDIMA', 'NDIMB', 'KWI', 'VWM', 'VWN', 'STRM', 'STRN', 'SA', 'SB']
    dtype = {x: np.float for x in feature_list}
    dataset = Dataset.from_dtype(
        name="sgemm",
        task="regression",
        label_column='runs_avg',
        dtype=dtype,
        drop_columns=['Run1 (ms)', 'Run2 (ms)', 'Run3 (ms)', 'Run4 (ms)']
    )
    return dataset.load_from_csv("sgemm.csv.gz", dtype=dtype)


def load_ovctt():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/Online+Video+Characteristics+and+Transcoding+Time+Dataset
    # directly read file transcoding_mesurment.tsv with pd.read_csv("..", sep="\t")

    feature_list = ['duration', 'width', 'height', 'bitrate', 'framerate', 'i', 'p', 'b', 'frames', 'i_size', 'p_size', 'b_size', 'size', 'o_bitrate', 'o_framerate', 'o_width', 'o_height', 'umem']
    dtype = {x: np.float for x in feature_list}
    dtype.update({'codec': "category", 'o_codec': "category"})
    dataset = Dataset.from_dtype(
        name="ovctt",
        task="regression",
        label_column='utime',
        dtype=dtype,
        drop_columns=['id']
    )
    return dataset.load_from_csv("ovctt.csv.gz", dtype=dtype)


def load_ypmsd():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/YearPredictionMSD
    # directly read and compressed with pandas

    dtype = {str(x): np.float for x in range(1, 90)}

    dataset = Dataset.from_dtype(
        name="ypmsd",
        task="regression",
        label_column='0',
        dtype=dtype,
    )
    return dataset.load_from_csv("yearpredictionmsd.csv.gz", dtype=dtype)

def load_nupvotes():
    # downloaded from https://www.kaggle.com/umairnsr87/predict-the-number-of-upvotes-a-post-will-get
    # only taken the train file which has labels

    dtype = {'Tag': "category",
             'Reputation': np.float,
             'Answers': np.float,
             'Views': np.float
             }

    dataset = Dataset.from_dtype(
        name="nupvotes",
        task="regression",
        label_column='Upvotes',
        dtype=dtype,
        drop_columns=['ID', 'Username'],
    )
    return dataset.load_from_csv("nupvotes.csv.gz", dtype=dtype)


def load_houseprices():
    # downloaded from https://www.kaggle.com/greenwing1985/housepricing?select=HousePrices_HalfMil.csv
    # SYNTHETIC DATASET
    # directly read and compressed

    dtype = {'Area':np.float,
             'Garage': "category",
             'FirePlace': "category",
             'Baths': np.float,
             'White Marble': "category",
             'Black Marble': "category",
             'Indian Marble': "category",
             'Floors': np.float,
             'City': "category",
             'Solar': "category",
             'Electric': "category",
             'Fiber': "category",
             'Glass Doors': "category",
             'Swiming Pool': "category",
             'Garden': "category"
             }

    dataset = Dataset.from_dtype(
        name="houseprices",
        task="regression",
        label_column='Prices',
        dtype=dtype,
    )
    return dataset.load_from_csv("houseprices.csv.gz", dtype=dtype)


def load_fifa19():
    # downloaded from https://www.kaggle.com/karangadiya/fifa19
    # preprocessed with fifa19_preprocess.py based on https://www.kaggle.com/nitindatta/fifa-in-depth-analysis-with-linear-regression/notebook

    features = ['Age', 'Potential', 'International Reputation', 'Weak Foot', 'Skill Moves', 'Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']

    dtype = {x: np.float for x in features}
    dtype.update({x: "category" for x in ['Real_Face', 'Right_Foot', 'Simple_Position', 'Major_Nation', 'WorkRate1', 'WorkRate2']})

    dataset = Dataset.from_dtype(
        name="fifa19",
        task="regression",
        label_column='Overall',
        dtype=dtype,
    )
    return dataset.load_from_csv("fifa19.csv.gz", dtype=dtype)


def load_madelon():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/Madelon
    # preprocessed with parser.py in data/madelon

    features = [str(i) for i in range(500)] #+ ["label"]


    dtype = {x: np.float for x in features}
    #dtype["label"] = "category"

    dataset = Dataset.from_dtype(
        name="madelon",
        task="binary-classification",
        label_column='label',
        dtype=dtype,
    )
    return dataset.load_from_csv("madelon.csv.gz", dtype=dtype)


def load_arcene():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/Arcene
    # preprocessed with parser.py in data/arcene

    features = [str(i) for i in range(10000)] #+ ["label"]


    dtype = {x: np.float for x in features}
    #dtype["label"] = "category"

    dataset = Dataset.from_dtype(
        name="arcene",
        task="binary-classification",
        label_column='label',
        dtype=dtype,
    )
    return dataset.load_from_csv("arcene.csv.gz", dtype=dtype)


def load_gene_expression():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/gene+expression+cancer+RNA-Seq
    # preprocessed with parser.py in data/TCGA-PANCAN-HiSeq-801x20531

    features = ["gene_"+str(i) for i in range(20531)] #+ ["label"]

    dtype = {x: np.float for x in features}
    #dtype["label"] = "category"

    dataset = Dataset.from_dtype(
        name="gene-expression",
        task="multiclass-classification",
        label_column='Class',
        dtype=dtype,
    )
    return dataset.load_from_csv("gene_expression.csv.gz", dtype=dtype)


def load_glaucoma():
    # directly downloaded from https://www.machinelearningplus.com/machine-learning/feature-selection/

    features = ['ag', 'at', 'as', 'an', 'ai', 'eag', 'eat', 'eas', 'ean', 'eai', 'abrg',
       'abrt', 'abrs', 'abrn', 'abri', 'hic', 'mhcg', 'mhct', 'mhcs', 'mhcn',
       'mhci', 'phcg', 'phct', 'phcs', 'phcn', 'phci', 'hvc', 'vbsg', 'vbst',
       'vbss', 'vbsn', 'vbsi', 'vasg', 'vast', 'vass', 'vasn', 'vasi', 'vbrg',
       'vbrt', 'vbrs', 'vbrn', 'vbri', 'varg', 'vart', 'vars', 'varn', 'vari',
       'mdg', 'mdt', 'mds', 'mdn', 'mdi', 'tmg', 'tmt', 'tms', 'tmn', 'tmi',
       'mr', 'rnf', 'mdic', 'emd', 'mv']

    dtype = {x: np.float for x in features}

    dataset = Dataset.from_dtype(
        name="glaucoma",
        task="binary-classification",
        label_column='Class',
        dtype=dtype,
    )
    return dataset.load_from_csv("glaucoma.csv.gz", dtype=dtype)

def load_gisette():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/Gisette
    # preprocessed with parser.py in data/gisette

    features = [str(i) for i in range(5000)]

    dtype = {x: np.float for x in features}
    #dtype["label"] = "category"

    dataset = Dataset.from_dtype(
        name="gisette",
        task="binary-classification",
        label_column='label',
        dtype=dtype,
    )
    return dataset.load_from_csv("gisette.csv.gz", dtype=dtype)



def load_amazon():
    # downloaded from https://www.openml.org/search?type=data&sort=runs&id=4135&status=active
    # preprocessed with parser.py in data/amazon

    dtype = {x: "category" for x in ['RESOURCE', 'MGR_ID', 'ROLE_ROLLUP_1', 'ROLE_ROLLUP_2', 'ROLE_DEPTNAME',
       'ROLE_TITLE', 'ROLE_FAMILY_DESC', 'ROLE_FAMILY', 'ROLE_CODE']}

    dataset = Dataset.from_dtype(
        name="amazon",
        task="binary-classification",
        label_column='target',
        dtype=dtype,
    )
    return dataset.load_from_csv("amazon.csv.gz", dtype=dtype)

def load_atp1d():
    # downloaded from https://www.openml.org/search?type=data&sort=runs&id=41475&status=active
    # preprocessed with parser.py in data/atp (change filenames in script  1d/7d)
    module_path = os.path.dirname(__file__)

    csv = pd.read_csv(os.path.join(module_path, "data", "atp1d.csv.gz"))
    columns = list(csv.columns)
    del csv
    label_names = []
    for j in range(6):
        label_names.append(columns.pop(-1))
    dtype = {x: np.float for x in columns}

    dataset = Dataset.from_dtype(
        name="atp1d",
        task="regression",
        label_column=label_names[-1], # We select only the first task and drop the rest of the labels
        drop_columns=label_names[:-1],
        dtype=dtype,
    )
    return dataset.load_from_csv("atp1d.csv.gz", dtype=dtype)


def load_atp7d():
    # downloaded from https://www.openml.org/search?type=data&sort=runs&id=41476&status=active
    # preprocessed with parser.py in data/atp (change filenames in script  1d/7d)
    module_path = os.path.dirname(__file__)

    csv = pd.read_csv(os.path.join(module_path, "data", "atp7d.csv.gz"))
    columns = list(csv.columns)
    del csv
    label_names = []
    for j in range(6):
        label_names.append(columns.pop(-1))
    dtype = {x: np.float for x in columns}

    dataset = Dataset.from_dtype(
        name="atp7d",
        task="regression",
        label_column=label_names[-1], # We select only the first task and drop the rest of the labels
        drop_columns=label_names[:-1],
        dtype=dtype,
    )
    return dataset.load_from_csv("atp7d.csv.gz", dtype=dtype)

def load_parkinson():
    # downloaded from https://www.openml.org/search?type=data&sort=runs&id=42176&status=active
    # preprocessed with parser.py in data/parkinson
    module_path = os.path.dirname(__file__)

    csv = pd.read_csv(os.path.join(module_path, "data", "parkinson.csv.gz"))
    columns = list(csv.columns)
    del csv

    dtype = {x: np.float for x in columns[2:-1]}
    dtype[columns[1]] = "category"

    dataset = Dataset.from_dtype(
        name="parkinson",
        task="binary-classification",
        label_column=columns[-1],
        drop_columns=columns[:1],
        dtype=dtype,
    )
    return dataset.load_from_csv("parkinson.csv.gz", dtype=dtype)

def load_gina_prior():
    # downloaded from https://www.openml.org/search?type=data&sort=runs&id=1042&status=active
    # preprocessed with parser.py in data/gina_prior

    dtype = {"pixel"+str(x+1): np.float for x in range(784)}

    dataset = Dataset.from_dtype(
        name="gina_prior",
        task="binary-classification",
        label_column="label",
        dtype=dtype,
    )
    return dataset.load_from_csv("gina_prior.csv.gz", dtype=dtype)

def load_gina():
    # downloaded from https://www.openml.org/search?type=data&status=active&id=41158
    # preprocessed with parser.py in data/gina

    dtype = {"V"+str(x+1): np.float for x in range(970)}

    dataset = Dataset.from_dtype(
        name="gina",
        task="binary-classification",
        label_column="class",
        dtype=dtype,
    )
    return dataset.load_from_csv("gina.csv.gz", dtype=dtype)

def load_qsar():
    # downloaded from https://www.openml.org/search?type=data&sort=runs&id=3987&status=active
    # preprocessed with parser.py in data/qsar

    dtype = {"FCFP6_1024_"+str(x): np.float for x in range(1024)}

    dataset = Dataset.from_dtype(
        name="qsar",
        task="regression",
        label_column="MEDIAN_PXC50",
        dtype=dtype,
    )
    return dataset.load_from_csv("qsartid11678.csv.gz", dtype=dtype)

def load_qsar10980():
    # downloaded from https://www.openml.org/search?type=data&status=active&id=3277
    # preprocessed with parser.py in data/qsar10980

    dtype = {"FCFP6_1024_"+str(x): np.float for x in range(1024)}

    dataset = Dataset.from_dtype(
        name="qsar10980",
        task="regression",
        label_column="MEDIAN_PXC50",
        dtype=dtype,
    )
    return dataset.load_from_csv("qsartid10980.csv.gz", dtype=dtype)

def load_santander():
    # downloaded from https://www.openml.org/search?type=data&status=active&id=42572
    # preprocessed with parser.py in data/santander
    module_path = os.path.dirname(__file__)

    csv = pd.read_csv(os.path.join(module_path, "data", "santander.csv.gz"))
    columns = list(csv.columns)
    del csv

    dtype = {x: np.float for x in columns[:-1]}

    dataset = Dataset.from_dtype(
        name="santander",
        task="regression",
        label_column=columns[-1],
        dtype=dtype,
    )
    return dataset.load_from_csv("santander.csv.gz", dtype=dtype)

def load_ap_colon_kidney():
    # downloaded from https://www.openml.org/search?type=data&sort=runs&id=1137&status=active
    # preprocessed with parser.py
    module_path = os.path.dirname(__file__)

    csv = pd.read_csv(os.path.join(module_path, "data", "ap_colon_kidney.csv.gz"))
    columns = list(csv.columns)
    del csv

    dtype = {x: np.float for x in columns[1:-1]}

    dataset = Dataset.from_dtype(
        name="ap_colon_kidney",
        task="binary_classification",
        label_column=columns[-1],
        drop_columns=columns[:1],
        dtype=dtype,
    )
    return dataset.load_from_csv("ap_colon_kidney.csv.gz", dtype=dtype)

def load_robert():
    # downloaded from https://www.openml.org/search?type=data&sort=runs&id=41165&status=active
    # preprocessed with parser.py

    dtype = {"V"+str(x+1): np.float for x in range(7200)}

    dataset = Dataset.from_dtype(
        name="robert",
        task="multiclass-classification",
        label_column="class",
        dtype=dtype,
    )
    return dataset.load_from_csv("robert.csv.gz", dtype=dtype)

def load_bioresponse():
    # downloaded from https://www.openml.org/search?type=data&sort=runs&id=4134&status=active
    # preprocessed with parser.py

    dtype = {"D"+str(x+1): np.float for x in range(1776)}

    dataset = Dataset.from_dtype(
        name="bioreponse",
        task="binary-classification",
        label_column="target",
        dtype=dtype,
    )
    return dataset.load_from_csv("bioresponse.csv.gz", dtype=dtype)

def load_christine():
    # downloaded from https://www.openml.org/search?type=data&sort=runs&id=41142&status=active
    # preprocessed with parser.py

    dtype = {"V"+str(x+1): np.float for x in range(1636)}

    dataset = Dataset.from_dtype(
        name="christine",
        task="binary-classification",
        label_column="class",
        dtype=dtype,
    )
    return dataset.load_from_csv("christine.csv.gz", dtype=dtype)

def load_hiva_agnostic():
    # downloaded from https://www.openml.org/search?type=data&sort=runs&id=1039&status=active
    # preprocessed with parser.py

    dtype = {"attr"+str(x): np.float for x in range(1617)}

    dataset = Dataset.from_dtype(
        name="hiva_agnostic",
        task="binary-classification",
        label_column="label",
        dtype=dtype,
    )
    return dataset.load_from_csv("hiva_agnostic.csv.gz", dtype=dtype)

def load_gpositivego():
    # downloaded from https://www.uco.es/kdis/mllresources/
    # preprocessed with parser.py

    dtype = {str(x): np.float for x in range(912)}

    dataset = Dataset.from_dtype(
        name="gpositivego",
        task="multiclass-classification",
        label_column="912",
        dtype=dtype,
    )
    return dataset.load_from_csv("gpositivego.csv.gz", dtype=dtype)

def load_gnegativego():
    # downloaded from https://www.uco.es/kdis/mllresources/
    # preprocessed with parser.py

    dtype = {str(x): np.float for x in range(1717)}

    dataset = Dataset.from_dtype(
        name="gnegativego",
        task="multiclass-classification",
        label_column="label",
        dtype=dtype,
    )
    return dataset.load_from_csv("gnegativego.csv.gz", dtype=dtype)

def load_gpositivepseaac():
    # downloaded from https://www.uco.es/kdis/mllresources/
    # preprocessed with parser.py

    dtype = {str(x): np.float for x in range(440)}

    dataset = Dataset.from_dtype(
        name="gpositivepseaac",
        task="multiclass-classification",
        label_column="label",
        dtype=dtype,
    )
    return dataset.load_from_csv("gpositivepseaac.csv.gz", dtype=dtype)


def load_gnegativepseaac():
    # downloaded from https://www.uco.es/kdis/mllresources/
    # preprocessed with parser.py

    dtype = {str(x): np.float for x in range(440)}

    dataset = Dataset.from_dtype(
        name="gnegativepseaac",
        task="multiclass-classification",
        label_column="label",
        dtype=dtype,
    )
    return dataset.load_from_csv("gnegativepseaac.csv.gz", dtype=dtype)



def load_nyctaxi():
    # downloaded from https://www.kaggle.com/c/nyc-taxi-trip-duration/data?select=test.zip
    # only using train file which has labels, preprocessed with nyctaxi_preprocess.py based on
    # https://www.kaggle.com/stephaniestallworth/nyc-taxi-eda-regression-fivethirtyeight-viz/notebook

    features = ['passenger_count', 'pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude', 'pickup_month', 'pickup_hour', 'pickup_weekday']

    dtype = {x: np.float for x in features}
    dtype.update({x: "category" for x in ['vendor_id', 'store_and_fwd_flag']})

    dataset = Dataset.from_dtype(
        name="nyctaxi",
        task="regression",
        label_column='trip_duration',
        dtype=dtype,
        drop_columns=["pickup_datetime", 'dropoff_datetime', 'pickup_date', 'pickup_time', 'dropoff_date', 'dropoff_time', 'id']
    )
    return dataset.load_from_csv("nyctaxi.csv.gz", dtype=dtype)



def load_wine():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/wine+quality
    # concatenated winequality-red/white into single dataframe

    features = ['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar', 'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

    dtype = {x: np.float for x in features}

    dataset = Dataset.from_dtype(
        name="wine",
        task="regression",
        label_column='quality',
        dtype=dtype,
    )
    return dataset.load_from_csv("wine.csv.gz", dtype=dtype)


def load_airbnb():
    # downloaded from https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data
    # preprocessed using preprocess_airbnb.py based on https://www.kaggle.com/xinjianzhao/airbnb-new-york-analysis-plus/notebook

    features = ['latitude', 'longitude', 'minimum_nights', 'reviews_count', 'reviews_per_month', 'host_listings_count', 'availability_365']

    dtype = {x: np.float for x in features}
    dtype.update({x: "category" for x in ['neighbourhood_group', 'room_type']})

    dataset = Dataset.from_dtype(
        name="airbnb",
        task="regression",
        label_column='price_log',
        dtype=dtype,
        drop_columns=['last_review', 'price', 'neighbourhood', 'index', 'id', 'name', 'host_name', 'host_id']
    )
    return dataset.load_from_csv("airbnb.csv.gz", dtype=dtype)





def load_epsilon_catboost():
    from catboost.datasets import epsilon

    df_train, df_test = epsilon()
    columns = list(df_train.columns)
    dataset = Dataset(
        name="epsilon",
        task="binary-classification",
        label_column=columns[0],
        continuous_columns=columns[1:],
        categorical_columns=None,
    )

    df = pd.concat([df_train, df_test], axis="index")
    dataset.df_raw = df
    return dataset


def load_covtype():
    from sklearn.datasets import fetch_covtype

    data = fetch_covtype(as_frame=True)
    df = data["frame"]
    continuous_columns = [col for col in df.columns if col != "Cover_Type"]
    categorical_columns = None
    dataset = Dataset(
        name="covtype",
        task="multiclass-classification",
        label_column="Cover_Type",
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
    )
    dataset.df_raw = df
    return dataset


def load_diabetes():
    from sklearn.datasets import load_diabetes

    data = load_diabetes(as_frame=True)
    df = data["frame"]
    continuous_columns = [col for col in df.columns if col != "target"]
    categorical_columns = []
    dataset = Dataset(
        name="diabetes",
        task="regression",
        label_column="target",
        continuous_columns=continuous_columns,
        categorical_columns=categorical_columns,
    )
    dataset.df_raw = df
    return dataset


def load_kddcup99():
    from sklearn.datasets import fetch_kddcup99

    # We load the full datasets with 4.8 million rows
    data = fetch_kddcup99(as_frame=True, percent10=False)
    df = data["frame"]
    # We change the dtypes (for some weird reason everything is "object"...)
    dtype = {
        "duration": np.float32,
        "protocol_type": "category",
        "service": "category",
        "flag": "category",
        "src_bytes": np.float32,
        "dst_bytes": np.float32,
        "land": "category",
        "wrong_fragment": np.float32,
        "urgent": np.float32,
        "hot": np.float32,
        "num_failed_logins": np.float32,
        "logged_in": "category",
        "num_compromised": np.float32,
        "root_shell": "category",
        "su_attempted": "category",
        "num_root": np.float32,
        "num_file_creations": np.float32,
        "num_shells": np.float32,
        "num_access_files": np.float32,
        "num_outbound_cmds": np.float32,
        "is_host_login": "category",
        "is_guest_login": "category",
        "count": np.float32,
        "srv_count": np.float32,
        "serror_rate": np.float32,
        "srv_serror_rate": np.float32,
        "rerror_rate": np.float32,
        "srv_rerror_rate": np.float32,
        "same_srv_rate": np.float32,
        "diff_srv_rate": np.float32,
        "srv_diff_host_rate": np.float32,
        "dst_host_count": np.float32,
        "dst_host_srv_count": np.float32,
        "dst_host_same_srv_rate": np.float32,
        "dst_host_diff_srv_rate": np.float32,
        "dst_host_same_src_port_rate": np.float32,
        "dst_host_srv_diff_host_rate": np.float32,
        "dst_host_serror_rate": np.float32,
        "dst_host_srv_serror_rate": np.float32,
        "dst_host_rerror_rate": np.float32,
        "dst_host_srv_rerror_rate": np.float32,
    }
    df = df.astype(dtype)
    dataset = Dataset.from_dtype(
        name="kddcup",
        task="multiclass-classification",
        label_column="labels",
        dtype=dtype,
    )
    dataset.df_raw = df
    return dataset


def load_letter():
    dtype = {
        "X0": np.float,
        "X1": np.float,
        "X2": np.float,
        "X3": np.float,
        "X4": np.float,
        "X5": np.float,
        "X6": np.float,
        "X7": np.float,
        "X8": np.float,
        "X9": np.float,
        "X10": np.float,
        "X11": np.float,
        "X12": np.float,
        "X13": np.float,
        "X14": np.float,
        "X15": np.float,
    }
    dataset = Dataset.from_dtype(
        name="letter",
        task="multiclass-classification",
        label_column="y",
        dtype=dtype,
        drop_columns=["Unnamed: 0"],
    )
    return dataset.load_from_csv("letter.csv.gz", dtype=dtype)


def load_satimage():
    dtype = {
        "X0": np.float,
        "X1": np.float,
        "X2": np.float,
        "X3": np.float,
        "X4": np.float,
        "X5": np.float,
        "X6": np.float,
        "X7": np.float,
        "X8": np.float,
        "X9": np.float,
        "X10": np.float,
        "X11": np.float,
        "X12": np.float,
        "X13": np.float,
        "X14": np.float,
        "X15": np.float,
        "X16": np.float,
        "X17": np.float,
        "X18": np.float,
        "X19": np.float,
        "X20": np.float,
        "X21": np.float,
        "X22": np.float,
        "X23": np.float,
        "X24": np.float,
        "X25": np.float,
        "X26": np.float,
        "X27": np.float,
        "X28": np.float,
        "X29": np.float,
        "X30": np.float,
        "X31": np.float,
        "X32": np.float,
        "X33": np.float,
        "X34": np.float,
        "X35": np.float,
    }
    dataset = Dataset.from_dtype(
        name="satimage",
        task="multiclass-classification",
        label_column="y",
        drop_columns=["Unnamed: 0"],
        dtype=dtype,
    )
    return dataset.load_from_csv("satimage.csv.gz", dtype=dtype)

def load_statlog():
    # downloaded from https://archive.ics.uci.edu/ml/datasets/Statlog+%28Landsat+Satellite%29
    dtype = {str(x): np.float for x in range(36)}
    dataset = Dataset.from_dtype(
        name="statlog",
        task="multiclass-classification",
        label_column="36",
        dtype=dtype,
    )
    return dataset.load_from_csv("statlog.csv.gz", dtype=dtype)


def load_mnist():
    from sklearn.datasets import fetch_openml

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=True)
    df = X.join(y)

    dtype = {"pixel" + str(i + 1): np.float for i in range(784)}

    dataset = Dataset.from_dtype(
        name="mnist",
        task="multiclass-classification",
        label_column="class",
        dtype=dtype,
    )
    return dataset.from_dataframe(df)


def load_iris():

    from sklearn.datasets import load_iris as ld_iris

    X, y = ld_iris(return_X_y=True, as_frame=True)
    df = X.join(y)
    dtype = {c: np.float for c in X.columns}

    dataset = Dataset.from_dtype(
        name="iris",
        task="multiclass-classification",
        label_column="target",
        dtype=dtype,
    )
    return dataset.from_dataframe(df)


def load_sensorless():
    dtype = {
        0: np.float,
        1: np.float,
        2: np.float,
        3: np.float,
        4: np.float,
        5: np.float,
        6: np.float,
        7: np.float,
        8: np.float,
        9: np.float,
        10: np.float,
        11: np.float,
        12: np.float,
        13: np.float,
        14: np.float,
        15: np.float,
        16: np.float,
        17: np.float,
        18: np.float,
        19: np.float,
        20: np.float,
        21: np.float,
        22: np.float,
        23: np.float,
        24: np.float,
        25: np.float,
        26: np.float,
        27: np.float,
        28: np.float,
        29: np.float,
        30: np.float,
        31: np.float,
        32: np.float,
        33: np.float,
        34: np.float,
        35: np.float,
        36: np.float,
        37: np.float,
        38: np.float,
        39: np.float,
        40: np.float,
        41: np.float,
        42: np.float,
        43: np.float,
        44: np.float,
        45: np.float,
        46: np.float,
        47: np.float,
    }
    dataset = Dataset.from_dtype(
        name="sensorless",
        task="multiclass-classification",
        label_column=48,
        dtype=dtype,
    )
    return dataset.load_from_csv("sensorless.csv.gz", sep=" ", header=None, dtype=dtype)


def load_spambase():
    dtype = {
        0: np.float,
        1: np.float,
        2: np.float,
        3: np.float,
        4: np.float,
        5: np.float,
        6: np.float,
        7: np.float,
        8: np.float,
        9: np.float,
        10: np.float,
        11: np.float,
        12: np.float,
        13: np.float,
        14: np.float,
        15: np.float,
        16: np.float,
        17: np.float,
        18: np.float,
        19: np.float,
        20: np.float,
        21: np.float,
        22: np.float,
        23: np.float,
        24: np.float,
        25: np.float,
        26: np.float,
        27: np.float,
        28: np.float,
        29: np.float,
        30: np.float,
        31: np.float,
        32: np.float,
        33: np.float,
        34: np.float,
        35: np.float,
        36: np.float,
        37: np.float,
        38: np.float,
        39: np.float,
        40: np.float,
        41: np.float,
        42: np.float,
        43: np.float,
        44: np.float,
        45: np.float,
        46: np.float,
        47: np.float,
        48: np.float,
        49: np.float,
        50: np.float,
        51: np.float,
        52: np.float,
        53: np.float,
        54: np.float,
        55: np.int,
        56: np.int,
    }
    dataset = Dataset.from_dtype(
        name="spambase",
        task="binary-classification",
        label_column=57,
        dtype=dtype,
    )
    return dataset.load_from_csv("spambase.csv.gz", header=None, dtype=dtype)


#


# class KDDCup(Datasets):  # multiclass
#     def __init__(
#         self,
#         path=None,
#         test_split=0.3,
#         random_state=0,
#         normalize_intervals=False,
#         one_hot_categoricals=False,
#         as_pandas=False,
#         subsample=None,
#     ):
#         from sklearn.datasets import fetch_kddcup99
#
#         print("Loading full KDDCdup datasets (percent10=False)")
#         print("")
#         data, target = fetch_kddcup99(
#             percent10=False, return_X_y=True, random_state=random_state, as_frame=True
#         )  # as_pandas)
#
#         if subsample is not None:
#             print("Subsampling datasets with subsample={}".format(subsample))
#
#         data = data[:subsample]
#         target = target[:subsample]
#
#         discrete = [
#             "protocol_type",
#             "service",
#             "flag",
#             "land",
#             "logged_in",
#             "is_host_login",
#             "is_guest_login",
#         ]
#         continuous = list(set(data.columns) - set(discrete))
#
#         dummies = pd.get_dummies(target)
#         dummies.columns = list(range(len(dummies.columns)))
#         self.target = dummies.idxmax(axis=1)  # .values
#         self.binary = False
#         self.task = "classification"
#
#         self.n_classes = self.target.max() + 1  # 23
#
#         X_continuous = data[continuous].astype("float32")
#         if normalize_intervals:
#             mins = X_continuous.min()
#             X_continuous = (X_continuous - mins) / (X_continuous.max() - mins)
#
#         if one_hot_categoricals:
#             X_discrete = pd.get_dummies(data[discrete], prefix_sep="#")  # .values
#         else:
#             # X_discrete = (data[discrete]).apply(lambda x: pd.factorize(x)[0])
#             X_discrete = data[discrete].apply(lambda x: pd.factorize(x)[0]).astype(int)
#
#         self.one_hot_categoricals = one_hot_categoricals
#         self.data = X_continuous.join(X_discrete)
#
#         if not as_pandas:
#             self.data = self.data.values
#             self.target = self.target.values
#         else:
#             self.data.columns = list(range(self.data.shape[1]))
#
#         self.size, self.n_features = self.data.shape
#         self.nb_continuous_features = len(continuous)  # 34#32
#
#         self.split_train_test(test_split, random_state)
#


# TODO: newsgroup is sparse, so we'll work on it later
def load_newsgroup():
    pass


# class NewsGroups(Datasets):  # multiclass
#     def __init__(
#         self,
#         path=None,
#         test_split=0.3,
#         random_state=0,
#         normalize_intervals=False,
#         one_hot_categoricals=False,
#         as_pandas=False,
#         subsample=None,
#     ):
#         from sklearn.datasets import fetch_20newsgroups_vectorized
#
#         data, target = fetch_20newsgroups_vectorized(
#             return_X_y=True, as_frame=True
#         )  # as_pandas)
#
#         if subsample is not None:
#             print("Subsampling datasets with subsample={}".format(subsample))
#
#         data = data[:subsample]
#         target = target[:subsample]
#
#         self.target = target
#         self.binary = False
#         self.task = "classification"
#
#         self.n_classes = self.target.max() + 1
#
#         if normalize_intervals:
#             mins = data.min()
#             data = (data - mins) / (data.max() - mins)
#
#         self.data = data
#
#         if not as_pandas:
#             self.data = self.data.values
#             self.target = self.target.values
#
#         self.size, self.n_features = self.data.shape
#         self.nb_continuous_features = self.n_features
#
#         self.split_train_test(test_split, random_state)


loaders_small_classification = [
    load_adult,
    load_bank,
    load_breastcancer,
    load_car,
    load_cardio,
    load_churn,
    load_default_cb,
    load_letter,
    load_satimage,
    load_sensorless,
    load_spambase,
    load_electrical,
    load_occupancy,
    load_avila,
]

loaders_small_regression = [load_boston, load_californiahousing, load_diabetes]

loaders_medium = [load_covtype]

loaders_large = []


def describe_datasets(include="small-classification", random_state=42):
    if include == "small-classification":
        loaders = loaders_small_classification
    elif include == "small-regression":
        loaders = loaders_small_regression
    else:
        raise ValueError("include=%r is not supported for now." % include)

    col_name = []
    col_n_samples = []
    col_n_features = []
    col_task = []
    col_n_classes = []
    col_n_features_categorical = []
    col_n_features_continuous = []
    col_scaled_gini = []
    col_n_samples_train = []
    col_n_samples_test = []
    col_n_columns = []
    for loader in loaders:
        dataset = loader()
        dataset.one_hot_encode = True
        dataset.standardize = True
        X_train, X_test, y_train, y_test = dataset.extract(random_state=random_state)
        n_samples_train, n_columns = X_train.shape
        n_samples_test, _ = X_test.shape
        col_name.append(dataset.name)
        col_task.append(dataset.task)
        col_n_samples.append(dataset.n_samples_)
        col_n_features.append(dataset.n_features_)
        col_n_classes.append(dataset.n_classes_)
        col_n_features_categorical.append(dataset.n_features_categorical_)
        col_n_features_continuous.append(dataset.n_features_continuous_)
        col_scaled_gini.append(dataset.scaled_gini_)
        col_n_samples_train.append(n_samples_train)
        col_n_samples_test.append(n_samples_test)
        col_n_columns.append(n_columns)

    if "regression" in include:
        df_description = pd.DataFrame(
            {
                "dataset": col_name,
                "task": col_task,
                "n_samples": col_n_samples,
                "n_samples_train": col_n_samples_train,
                "n_samples_test": col_n_samples_test,
                "n_features_cat": col_n_features_categorical,
                "n_features_cont": col_n_features_continuous,
                "n_features": col_n_features,
                "n_columns": col_n_columns,
            }
        )
    else:
        df_description = pd.DataFrame(
            {
                "dataset": col_name,
                "task": col_task,
                "n_samples": col_n_samples,
                "n_samples_train": col_n_samples_train,
                "n_samples_test": col_n_samples_test,
                "n_features_cat": col_n_features_categorical,
                "n_features_cont": col_n_features_continuous,
                "n_features": col_n_features,
                "n_classes": col_n_classes,
                "n_columns": col_n_columns,
                "scaled_gini": col_scaled_gini,
            }
        )

    return df_description


if __name__ == "__main__":
    # df_descriptions = describe_datasets()
    # print(df_descriptions)
    #
    # datasets = load_covtype()

    load_kddcup99()

    # print(datasets)
