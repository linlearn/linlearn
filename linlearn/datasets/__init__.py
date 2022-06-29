"""
This modules includes dataset loaders for experiments conducted with WildWood
"""

from .dataset import Dataset

from ._adult import load_adult
from ._bank import load_bank
from ._higgs import load_higgs
from ._car import load_car
from ._kick import load_kick
# from ._amazon import load_amazon
from ._epsilon import load_epsilon
from ._internet import load_internet

from .loaders import (
    load_boston,
    load_letor,
    load_epsilon_catboost,
    load_breastcancer,
    load_californiahousing,
    load_cardio,
    load_churn,
    load_covtype,
    load_diabetes,
    load_default_cb,
    load_iris,
    load_kddcup99,
    load_letter,
    load_mnist,
    load_satimage,
    load_sensorless,
    load_spambase,
    load_electrical,
    load_occupancy,
    load_avila,
    load_miniboone,
    load_gas,
    load_eeg,
    load_drybean,
    load_energy,
    load_bike,
    load_cbm,
    load_ccpp,
    load_gasturbine,
    load_casp,
    load_metro,
    load_superconduct,
    load_sgemm,
    load_ovctt,
    load_ypmsd,
    load_nupvotes,
    load_houseprices,
    load_fifa19,
    load_nyctaxi,
    load_wine,
    load_airbnb,
    load_statlog,
    load_madelon,
    load_arcene,
    load_gene_expression,
    load_glaucoma,
    load_gisette,
    load_amazon,
    load_atp1d,
    load_atp7d,
    load_gpositivego,
    load_gnegativego,
    load_gpositivepseaac,
    load_gnegativepseaac,
    load_parkinson,
    load_gina_prior,
    load_gina,
    load_qsar,
    load_qsar10980,
    load_santander,
    load_ap_colon_kidney,
    load_robert,
    load_bioresponse,
    load_christine,
    load_hiva_agnostic,
    describe_datasets,
)

from .signals import get_signal, make_regression
