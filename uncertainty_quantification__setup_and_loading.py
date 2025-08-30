import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import random
import torch
import pandas as pd
from tqdm import tqdm
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import math
import itertools
import os
import pickle
import json
from pathlib import Path

prefix = '/home/3252395' # HPC account
import sys
sys.path.append(f'{prefix}/gtm/')

from corpus import GTMCorpus
from gtm import GTM

import warnings
warnings.filterwarnings("ignore", message=".*This DataLoader will create.*")
warnings.filterwarnings("ignore", category=FutureWarning)

# === Simulation Parameters ===
num_docs = 10000
num_topics = 2
vocab_size = 500
min_words = 100
max_words = 100
num_covs = 1
num_iter = 1000
patience = 75
num_epochs = 75

doc_topic_prior = 'logistic_normal'
lambda_ = np.array([[0, 0], [0, -0.5]])
beta = np.array([0, -0.5])
sigma = np.array([[1.7, -0.3], [-0.3, 2.3]])
sigma_y = 1

# Mode
# mode = 'upstream'
# mode = 'downstream'
mode = 'both'

# === Utility ===
def match_columns(estimated_df, true_df):
    score_list = []
    for true_col in true_df.columns:
        true_target_col = true_df[true_col]
        score_list_per_row = [np.dot(estimated_df[col], true_target_col) for col in estimated_df.columns]
        score_list.append(score_list_per_row)
    score_matrix = pd.DataFrame(score_list)
    true_topics, estimated_topics = linear_sum_assignment(-score_matrix)
    mapping = {f"Topic{t}": f"Topic{e}" for t, e in zip(true_topics, estimated_topics)}
    inverse = {t: e for t, e in zip(true_topics, estimated_topics)}
    df_matched = estimated_df.loc[:, mapping.values()]
    df_matched.columns = mapping.keys()
    return df_matched, inverse

# Recreate the same suffix string used when saving:
enc, dec, w_prior, w_pred_loss = [64], [], 1, 1
suffix = f'mode={mode}_docs={num_docs}_iter={num_iter}_patience={patience}_epochs={num_epochs}__enc={enc}_dec={dec}_w_prior={w_prior}_w_pred_loss={w_pred_loss}'

# Point to the same existing folder
ITER_SAVE_DIR = os.path.join(prefix, "iter_artifacts")

def load_iter_artifacts(iter_idx, suffix, save_dir=ITER_SAVE_DIR, model_kind="joint", GTM_cls=None):
    """
    Load dataset and ONE model ('joint' or 'two_stage') for a given iteration.
    Returns: (dataset_payload, model_obj_or_tuple)
      - dataset_payload is the dict saved in *_dataset.pkl
      - model is the restored GTM object
    """
    # --- dataset ---
    dataset_path = os.path.join(save_dir, f"{suffix}__iteration_{iter_idx}__dataset.pkl")
    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    # --- model meta ---
    meta_path = os.path.join(save_dir, f"{suffix}__iteration_{iter_idx}__model_meta.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)

    if model_kind not in meta["files"]:
        raise FileNotFoundError(f"No '{model_kind}' model recorded in meta for iteration {iter_idx}.")

    m = meta["files"][model_kind]
    mformat, mpath = m["format"], m["path"]

    # --- model ---
    try:
        with open(mpath, "rb") as f:
            model = pickle.load(f)
        return dataset, model
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {mpath} with format {mformat}: {e}")

def load_model_and_data(iteration_to_inspect, model_kind):
    dataset_payload, model_obj = load_iter_artifacts(
        iter_idx=iteration_to_inspect,
        suffix=suffix,
        save_dir=ITER_SAVE_DIR,
        model_kind=model_kind,
        GTM_cls=GTM,
    )
    df = dataset_payload["df"]
    df_true_dist_list = dataset_payload["df_true_dist_list"]
    true_df = dataset_payload["df_true_dist_list"][0]#["true_df"]
    params = dataset_payload["params"]

    # model_obj
    tm = model_obj

    if model_kind == "joint":
        data = GTMCorpus(df, prevalence="~ cov_1", labels="~ y - 1")
    else:
        data = GTMCorpus(df)

    return df, df_true_dist_list, true_df, params, tm, data

# === Load Data ===
iteration_to_load = 999
save_dir = os.path.join(prefix, "intermediate_results/upstream_and_downstream")
load_path = os.path.join(save_dir, suffix + f"__iteration_{iteration_to_load}.pkl")

with open(load_path, "rb") as f:
    data = pickle.load(f)

# Unpack variables
true_doc_topic_list = data["true_doc_topic_list"]
estimated_doc_topic_list_joint = data["estimated_doc_topic_list_joint"]
estimated_doc_topic_list_two_stage = data["estimated_doc_topic_list_two_stage"]
dict_upstream_joint = data["dict_upstream_joint"]
dict_upstream_two_stage = data["dict_upstream_two_stage"]
dict_upstream_infeasible = data["dict_upstream_infeasible"]
dict_downstream_joint = data["dict_downstream_joint"]
dict_downstream_two_stage = data["dict_downstream_two_stage"]
dict_downstream_infeasible = data["dict_downstream_infeasible"]
upstream_coverage = data["upstream_coverage"]
downstream_coverage = data["downstream_coverage"]

print(f"Loaded results from iteration {iteration_to_load}")
