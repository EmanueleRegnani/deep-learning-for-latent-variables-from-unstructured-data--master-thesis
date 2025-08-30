num_docs, num_topics = 10000, 2
vocab_size, min_words, max_words = 500, 100, 100
num_covs, doc_topic_prior = 0, 'dirichlet'

gtm_model_args = {
    "n_topics":num_topics, "print_topics":False,
    "update_prior":False, "doc_topic_prior":doc_topic_prior,
    "encoder_hidden_layers":[128], "decoder_hidden_layers":[],
    "print_every_n_epochs":10, "log_every_n_epochs":1000,
    "w_prior":1000,
    "patience":10, "learning_rate":1e-3, "batch_size":64, "num_epochs":150,
    "dropout":0, "device":None, "seed":42, "ckpt":None
    }

df_true_dist_list_gtm, df, anchor_words = generate_documents(
    num_docs, num_topics, vocab_size, num_covs=0,
    beta=None, lambda_=None, sigma_y=0.1, sigma_topic=None,
    doc_topic_prior=doc_topic_prior,
    min_words=min_words, max_words=min_words, anchor_words_per_topic=10,
    random_seed=42, include_y=False
)

full_dataset = GTMCorpus(df)
