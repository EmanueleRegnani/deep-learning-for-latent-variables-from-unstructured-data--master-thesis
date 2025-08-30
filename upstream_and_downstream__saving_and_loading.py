ITER_SAVE_DIR = os.path.join(prefix, "iter_artifacts")

def _model_save_joint_two_stage(iter_idx, tm_joint, tm_ts, suffix, save_dir=ITER_SAVE_DIR):
    """
    Save the trained joint and two-stage models for this iteration.
    Produces:
      <suffix>__iteration_<i>__joint.pkl
      <suffix>__iteration_<i>__two_stage.pkl
      <suffix>__iteration_<i>__model_meta.json
    """
    Path(save_dir).mkdir(parents=False, exist_ok=True)  # Fails if folder missing

    meta = {
        "suffix": str(suffix),
        "iteration": int(iter_idx),
        "files": {},
    }

    # --- JOINT ---
    joint_base = os.path.join(save_dir, f"{suffix}__iteration_{iter_idx}__joint")
    model_path = joint_base + ".pkl"
    with open(model_path, "wb") as f:
        pickle.dump(tm_joint, f)
    meta["files"]["joint"] = {"path": model_path, "format": "pickle"}

    # --- TWO-STAGE ---
    ts_base = os.path.join(save_dir, f"{suffix}__iteration_{iter_idx}__two_stage")
    model_path = ts_base + ".pkl"
    with open(model_path, "wb") as f:
        pickle.dump(tm_ts, f)
    meta["files"]["two_stage"] = {"path": model_path, "format": "pickle"}

    # Save small JSON with pointers/formats
    meta_path = os.path.join(save_dir, f"{suffix}__iteration_{iter_idx}__model_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    return meta_path


def _dataset_save(iter_idx, df, df_true_dist_list, true_df, params_dict, suffix, save_dir=ITER_SAVE_DIR):
    """
    Save the simulated dataset for this iteration.
    Produces:
      <suffix>__iteration_<i>__dataset.pkl
    Contents:
      df, df_true_dist_list, true_df, params (to reproduce)
    """
    Path(save_dir).mkdir(parents=False, exist_ok=True)
    dataset_path = os.path.join(save_dir, f"{suffix}__iteration_{iter_idx}__dataset.pkl")
    payload = {
        "df": df,
        "df_true_dist_list": df_true_dist_list,
        # "true_df": true_df,
        "params": params_dict,  # hyperparams and seeds used to generate/train
    }
    with open(dataset_path, "wb") as f:
        pickle.dump(payload, f)
    return dataset_path


load_from_intermediate_results = True
iteration_to_load = 999 # ignored if load_from_intermediate_results=False

if not load_from_intermediate_results:
    suffix = f'mode={mode}_docs={num_docs}_iter={num_iter}_patience={patience}_epochs={num_epochs}__enc={enc}_dec={dec}_w_prior={w_prior}_w_pred_loss={w_pred_loss}'
else:
    suffix = f'mode={mode}_docs={num_docs}_iter={num_iter}_patience={patience}_epochs={num_epochs}__enc={enc}_dec={dec}_w_prior={w_prior}_w_pred_loss={w_pred_loss}'

# -- Storage --
if not load_from_intermediate_results: # initialize anew
    start_iter = 0

    true_doc_topic_list = []
    estimated_doc_topic_list_joint = []
    estimated_doc_topic_list_two_stage = []

    dict_upstream_joint = {k: [] for k in range(num_topics)}
    dict_upstream_two_stage = {k: [] for k in range(num_topics)}
    dict_upstream_infeasible = {k: [] for k in range(num_topics)}

    dict_downstream_joint = {k: [] for k in range(num_topics)}
    dict_downstream_two_stage = {k: [] for k in range(num_topics)}
    dict_downstream_infeasible = {k: [] for k in range(num_topics)}

    upstream_coverage = {"joint": [0]*num_topics, "two_stage": [0]*num_topics, "infeasible": [0]*num_topics}
    downstream_coverage = {"joint": [0]*num_topics, "two_stage": [0]*num_topics, "infeasible": [0]*num_topics}
else:
    start_iter = iteration_to_load + 1

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

    print(f"Loaded intermediate results from iteration {iteration_to_load}", f"Length of intermediate results = {len(true_doc_topic_list)}\n\n", sep='\n')
    suffix = f'mode={mode}_docs={num_docs}_iter={num_iter}_patience={patience}_epochs={num_epochs}__enc={enc}_dec={dec}_w_prior={w_prior}_w_pred_loss={w_pred_loss}'


# === Monte Carlo Loop ===
for i in tqdm(range(num_iter)):
# ...
# ...
# ...

    # --- Save dataset + both models for THIS iteration ---
    params_for_repro = dict(
        random_seed=i, mode=mode,
        num_docs=num_docs, num_topics=num_topics,
        vocab_size=vocab_size, num_covs=num_covs,
        beta=beta, lambda_=lambda_, sigma_y=sigma_y, sigma_topic=sigma,
        doc_topic_prior=doc_topic_prior,
        min_words=min_words, max_words=max_words,
        enc=enc, dec=dec, w_prior=w_prior, w_pred_loss=w_pred_loss,
        patience=patience, num_epochs=num_epochs,
    )

    ds_path = _dataset_save(i, df=df, df_true_dist_list=df_true_dist_list, true_df=true_df,
                            params_dict=params_for_repro, suffix=suffix, save_dir=ITER_SAVE_DIR)
    meta_path = _model_save_joint_two_stage(i, tm_joint=tm_joint, tm_ts=tm_ts,
                                            suffix=suffix, save_dir=ITER_SAVE_DIR)
    print(f"\nIteration {i} - Saved dataset -> {ds_path}")
    print(f"Iteration {i} - Saved model meta -> {meta_path}\n")


    # Save partial results every 10 iterations
    if (i+1) % 10 == 0:
        save_dir = os.path.join(prefix, "intermediate_results/upstream_and_downstream")
        save_path = os.path.join(save_dir, suffix + f"__iteration_{i}.pkl")
        with open(save_path, "wb") as f:
            pickle.dump({
                "true_doc_topic_list": true_doc_topic_list,
                "estimated_doc_topic_list_joint": estimated_doc_topic_list_joint,
                "estimated_doc_topic_list_two_stage": estimated_doc_topic_list_two_stage,
                "dict_upstream_joint": dict_upstream_joint,
                "dict_upstream_two_stage": dict_upstream_two_stage,
                "dict_upstream_infeasible": dict_upstream_infeasible,
                "dict_downstream_joint": dict_downstream_joint,
                "dict_downstream_two_stage": dict_downstream_two_stage,
                "dict_downstream_infeasible": dict_downstream_infeasible,
                "upstream_coverage": upstream_coverage,
                "downstream_coverage": downstream_coverage
            }, f)

        print(f"\n\nSaved results up to iteration {i} to {save_path}", f"Length of results = {len(true_doc_topic_list)}", sep='\n')
