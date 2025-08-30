# === Simulation Parameters ===
num_docs, num_topics = 10000, 2
vocab_size, min_words, max_words = 500, 100, 100
num_covs = 1
num_iter = 10
patience, num_epochs = 75, 75
doc_topic_prior = 'logistic_normal'
lambda_ = np.array([[0, 0], [0, -0.5]])
beta = np.array([0, -0.5])
sigma = np.array([[1.7, -0.3], [-0.3, 2.3]])
sigma_y = 1

# Parameter Grid for Grid Search
encoder_layer_configs = [[128]#, [64], [256], [128, 64], [256, 128, 64]
]
decoder_layer_configs = [[]]
w_prior_configs = [1#, 0, 0.1, 10, 100, 1000, 5000
    ]
w_pred_loss_configs = [1#, 0, 0.1, 10, 100, 1000, 5000
    ]

# Mode
mode = 'both' # or 'upstream' or 'downstream'

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


# === Grid Search Loop ===
for enc, dec, w_prior, w_pred_loss in itertools.product(encoder_layer_configs, decoder_layer_configs, w_prior_configs, w_pred_loss_configs):
    print(f"\n\n=== Running grid search for enc={enc}, dec={dec}, w_prior={w_prior}, w_pred_loss={w_pred_loss} ===\n", end='-'*100)
    suffix = f'mode={mode}_docs={num_docs}_iter={num_iter}_patience={patience}_epochs={num_epochs}__enc={enc}_dec={dec}_w_prior={w_prior}_w_pred_loss={w_pred_loss}'

    # -- Storage --
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


    # === Monte Carlo Loop ===
    for i in tqdm(range(num_iter)):
        df_true_dist_list, df = generate_documents(
            num_docs=num_docs,
            num_topics=num_topics,
            vocab_size=vocab_size,
            num_covs=num_covs,
            beta=beta,
            lambda_=lambda_,
            sigma_y=sigma_y,
            sigma_topic=sigma,
            doc_topic_prior=doc_topic_prior,
            min_words=min_words,
            max_words=max_words,
            include_y=True,
            random_seed=i
        )

        true_df = df_true_dist_list[0]
        true_doc_topic_list.append(true_df)

        # -- GTM: Joint --
        print("\n\n  -- Training Joint Model\n")
        if mode == 'downstream':
            data_joint = GTMCorpus(df, labels ="~ y - 1")  # No covariates in prevalence model
            update_prior = False
            prevalence_model_type = 'RidgeCV'
            predictor_bias = False
            predictor_type = 'regressor'
        else: # i.e. 'upstream' or 'both'
            data_joint = GTMCorpus(df, prevalence="~ cov_1")
            update_prior = True
            prevalence_model_type = 'LinearRegression'
            predictor_bias = True
            predictor_type = None
        if mode == 'both': # in case it disagrees with 'upstream'
            data_joint = GTMCorpus(df, prevalence="~ cov_1", labels ="~ y - 1")
            predictor_bias = False
            predictor_type = "regressor"
        tm_joint = GTM(train_data=data_joint, n_topics=num_topics, update_prior=update_prior,
                      doc_topic_prior=doc_topic_prior, prevalence_model_type=prevalence_model_type,
                      encoder_hidden_layers=enc, decoder_hidden_layers=dec, w_prior=w_prior, w_pred_loss=w_pred_loss,
                      predictor_type=predictor_type, predictor_bias=predictor_bias, dropout=0,
                      learning_rate=1e-3, patience=patience, num_epochs=num_epochs, ckpt_folder=f"{prefix}/checkpoints/joint",
                      print_topics=False, print_every_n_epochs=5)

        doc_topic_joint = pd.DataFrame(tm_joint.get_doc_topic_distribution(data_joint),
                                      columns=[f"Topic{i}" for i in range(num_topics)])
        doc_topic_joint, map_joint = match_columns(doc_topic_joint, true_df)
        estimated_doc_topic_list_joint.append(doc_topic_joint)

        # -- GTM: Two-stage --
        print("\n\n  -- Training Two-stage Model\n")
        if mode == 'downstream':
            data_ts = GTMCorpus(df)  # No covariates in prevalence model
            update_prior = False
        else: # i.e. 'upstream' or 'both'
            data_ts = GTMCorpus(df, prevalence="~1")
            update_prior = True
        if mode == 'both': # in case it disagrees with 'upstream'
            data_ts = GTMCorpus(df)
            update_prior = False
        tm_ts = GTM(train_data=data_ts, n_topics=num_topics, update_prior=update_prior,
                    doc_topic_prior=doc_topic_prior,
                    encoder_hidden_layers=enc, decoder_hidden_layers=dec, w_prior=w_prior, w_pred_loss=w_pred_loss,
                    dropout=0, learning_rate=1e-3, patience=patience, num_epochs=num_epochs, ckpt_folder=f"{prefix}/checkpoints/two_stage",
                    print_topics=False, print_every_n_epochs=5)

        doc_topic_ts = pd.DataFrame(tm_ts.get_doc_topic_distribution(data_ts),
                                    columns=[f"Topic{i}" for i in range(num_topics)])
        doc_topic_ts, map_ts = match_columns(doc_topic_ts, true_df)
        estimated_doc_topic_list_two_stage.append(doc_topic_ts)

        if mode in ['upstream', 'both']:
            # -- Upstream: Topic k ~ cov_1 --
            print("\n\n  -- Computing Results for Upstream Model\n")
            X = sm.add_constant(df[["cov_1"]]).reset_index(drop=True)  # Ensure X has the correct indices
            for topic in range(num_topics):
                y_joint = np.log((doc_topic_joint[f"Topic{topic}"] + 1e-7) / (1 - doc_topic_joint[f"Topic{topic}"] + 1e-7)).reset_index(drop=True)
                y_ts = np.log((doc_topic_ts[f"Topic{topic}"] + 1e-7) / (1 - doc_topic_ts[f"Topic{topic}"] + 1e-7)).reset_index(drop=True)
                y_true = np.log((true_df[f"Topic{topic}"] + 1e-7) / (1 - true_df[f"Topic{topic}"] + 1e-7)).reset_index(drop=True)

                for y, name, store, cov in zip(
                    [y_joint, y_ts, y_true],
                    ["joint", "two_stage", "infeasible"],
                    [dict_upstream_joint, dict_upstream_two_stage, dict_upstream_infeasible],
                    [upstream_coverage["joint"], upstream_coverage["two_stage"], upstream_coverage["infeasible"]]
                ):
                    model = sm.OLS(y, X).fit(cov_type='HC1')
                    est = model.params[1]
                    ci = model.conf_int().loc["cov_1"]
                    store[topic].append(est)
                    if ci[0] <= lambda_[1, topic] <= ci[1]:
                        cov[topic] += 1

        if mode in ['downstream', 'both']:
            # -- Downstream: y ~ topic shares --
            print("\n\n  -- Computing Results for Downstream Model\n")
            y = df["y"].values
            for Z, name, store, cov in zip(
                [doc_topic_joint, doc_topic_ts, true_df],
                ["joint", "two_stage", "infeasible"],
                [dict_downstream_joint, dict_downstream_two_stage, dict_downstream_infeasible],
                [downstream_coverage["joint"], downstream_coverage["two_stage"], downstream_coverage["infeasible"]]
            ):
                model = sm.OLS(y, Z).fit(cov_type='HC1')
                for topic in range(num_topics):
                    est = model.params[topic]
                    ci = model.conf_int().loc[f"Topic{topic}"]
                    store[topic].append(est)
                    if ci[0] <= beta[topic] <= ci[1]:
                        cov[topic] += 1