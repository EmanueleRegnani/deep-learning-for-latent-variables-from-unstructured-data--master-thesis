encoder_layer_configs = [[128]#, [64], [256], [128, 64], [256, 128, 64] # grid search over encoder layers
]
decoder_layer_configs = [[]]
w_prior_configs = [1000#, 0, 0.1, 1, 10, 100, 1000, 5000 # grid search over w_prior
]

for enc in encoder_layer_configs:
  for dec in decoder_layer_configs:
    for w_prior in w_prior_configs:
      gtm_model_args['encoder_hidden_layers'] = enc
      gtm_model_args['decoder_hidden_layers'] = dec
      gtm_model_args['w_prior'] = w_prior
      gtm_model_args['ckpt'] = None # avoid loading checkpoints
      print(f"\nRunning GTM with encoder_hidden_layers={enc}, decoder_hidden_layers={dec}, and w_prior={w_prior}\n")
      tm = GTM(train_data=full_dataset, **gtm_model_args)

      df_doc_topic_gtm = pd.DataFrame(
          tm.get_doc_topic_distribution(full_dataset),
          index=[f"Doc{i}" for i in range(num_docs)],
          columns=[f"Topic{i}" for i in range(num_topics)],
      )

      estimated_df = df_doc_topic_gtm
      true_df = df_true_dist_list_gtm[0]

      # Matching topics
      score_matrix = np.array([
          [np.dot(estimated_df[f"Topic{e}"], true_df[f"Topic{t}"]) for e in range(num_topics)]
          for t in range(num_topics)
      ])
      true_topics, estimated_topics = linear_sum_assignment(-score_matrix)
      matched = {f"Topic{t}": f"Topic{e}" for t, e in zip(true_topics, estimated_topics)}
      reanged_df_gtm = estimated_df[[matched[f"Topic{i}"] for i in range(num_topics)]]
      reanged_df_gtm.columns = [f"Topic{i}" for i in range(num_topics)]