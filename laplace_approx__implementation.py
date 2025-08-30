from collections import defaultdict
from scipy.stats import norm, t

def laplace_linear_posterior(X, y, prior_var=np.inf):
    # ...
    # ...
    # ...
    
    return beta_hat, Sigma_N

# --- Laplace config (used only for drawing percentiles from the Gaussian posterior) ---
laplace_samples = 1000  # number of posterior draws for percentile CIs
alpha = 0.05            # 95% CI
z = norm.ppf(1 - alpha/2)

# --- containers for Laplace results ---
dict_upstream_joint_mc = defaultdict(list)
dict_upstream_two_stage_mc = defaultdict(list)

upstream_coverage_mc = {
    "joint": defaultdict(int),
    "two_stage": defaultdict(int),
    "joint_norm": defaultdict(int),
    "two_stage_norm": defaultdict(int),
}

dict_downstream_joint_mc = defaultdict(list)
dict_downstream_two_stage_mc = defaultdict(list)

downstream_coverage_mc = {
    "joint": defaultdict(int),
    "two_stage": defaultdict(int),
    "joint_norm": defaultdict(int),
    "two_stage_norm": defaultdict(int),
}

def _logit(p):
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)

store = {
    "upstream": {
        "percentile": {"joint":  [ [] for _ in range(num_topics) ],
                       "two_stage":[ [] for _ in range(num_topics) ]},
        "normal":     {"joint":  [ [] for _ in range(num_topics) ],
                       "two_stage":[ [] for _ in range(num_topics) ]},
    },
    "downstream": {
        "percentile": {"joint":  [ [] for _ in range(num_topics) ],
                       "two_stage":[ [] for _ in range(num_topics) ]},
        "normal":     {"joint":  [ [] for _ in range(num_topics) ],
                       "two_stage":[ [] for _ in range(num_topics) ]},
    },
}

q_lo, q_hi = 100 * (alpha/2), 100 * (1 - alpha/2)

for i in tqdm(range(num_iter)):
  # -- Load models and data --
  df, df_true_dist_list, true_df, params, tm_joint, data_joint = load_model_and_data(iteration_to_inspect=i, model_kind="joint")
  df, df_true_dist_list, true_df, params, tm_ts, data_ts = load_model_and_data(iteration_to_inspect=i, model_kind="two_stage")

  doc_topic_joint = pd.DataFrame(tm_joint.get_doc_topic_distribution(data_joint),
                                    columns=[f"Topic{k}" for k in range(num_topics)])
  doc_topic_joint, map_joint = match_columns(doc_topic_joint, true_df)

  doc_topic_ts = pd.DataFrame(tm_ts.get_doc_topic_distribution(data_ts),
                                columns=[f"Topic{k}" for k in range(num_topics)])
  doc_topic_ts, map_ts = match_columns(doc_topic_ts, true_df)

  X = sm.add_constant(df[["cov_1"]]).reset_index(drop=True)
  y = df["y"].values

  # === Upstream: Topic_k ~ cov_1 on logit scale (Laplace) ===
  for topic in range(num_topics):
      # Joint model
      y_joint = _logit(doc_topic_joint[f"Topic{topic}"].values)
      beta_hat_joint, Sigma_joint = laplace_linear_posterior(X, y_joint)

      # Two-stage model
      y_ts = _logit(doc_topic_ts[f"Topic{topic}"].values)
      beta_hat_ts, Sigma_ts = laplace_linear_posterior(X, y_ts)

      # Draws from Gaussian posterior for percentile CIs (slope is index 1)
      draws_joint = np.random.multivariate_normal(beta_hat_joint, Sigma_joint, size=laplace_samples)
      draws_ts    = np.random.multivariate_normal(beta_hat_ts,    Sigma_ts,    size=laplace_samples)

      joint_slopes = draws_joint[:, 1]
      ts_slopes    = draws_ts[:, 1]

      # ---- Save for plotting (percentile) ----
      store["upstream"]["percentile"]["joint"][topic].append(joint_slopes)
      store["upstream"]["percentile"]["two_stage"][topic].append(ts_slopes)

      # ---- Save for plotting (normal) ----
      m_joint, s_joint = beta_hat_joint[1], np.sqrt(Sigma_joint[1, 1])
      m_ts,    s_ts    = beta_hat_ts[1],    np.sqrt(Sigma_ts[1, 1])
      store["upstream"]["normal"]["joint"][topic].append((m_joint, s_joint))
      store["upstream"]["normal"]["two_stage"][topic].append((m_ts,    s_ts))

      # ---- Point estimates (posterior means / MAP under flat prior) ----
      dict_upstream_joint_mc[topic].append(m_joint)
      dict_upstream_two_stage_mc[topic].append(m_ts)

      # ---- Percentile CIs (from draws) + coverage vs true lambda ----
      ci_joint_mc = np.percentile(joint_slopes, [q_lo, q_hi])
      ci_ts_mc    = np.percentile(ts_slopes,    [q_lo, q_hi])

      if ci_joint_mc[0] <= lambda_[1, topic] <= ci_joint_mc[1]:
          upstream_coverage_mc["joint"][topic] += 1
      if ci_ts_mc[0] <= lambda_[1, topic] <= ci_ts_mc[1]:
          upstream_coverage_mc["two_stage"][topic] += 1

      # ---- Normal-based CIs (mean ± z*sd) + coverage ----
      ci_joint_norm = np.array([m_joint - z*s_joint, m_joint + z*s_joint])
      ci_ts_norm    = np.array([m_ts    - z*s_ts,    m_ts    + z*s_ts])

      if ci_joint_norm[0] <= lambda_[1, topic] <= ci_joint_norm[1]:
          upstream_coverage_mc["joint_norm"][topic] = upstream_coverage_mc["joint_norm"].get(topic, 0) + 1
      if ci_ts_norm[0] <= lambda_[1, topic] <= ci_ts_norm[1]:
          upstream_coverage_mc["two_stage_norm"][topic] = upstream_coverage_mc["two_stage_norm"].get(topic, 0) + 1

  # === Downstream: y ~ topic shares (Laplace) ===
  # Joint
  Z_joint = doc_topic_joint.values  # (n_docs, K)
  beta_hat_joint, Sigma_joint = laplace_linear_posterior(Z_joint, y)
  draws_joint = np.random.multivariate_normal(beta_hat_joint, Sigma_joint, size=laplace_samples)

  # Two-stage
  Z_ts = doc_topic_ts.values
  beta_hat_ts, Sigma_ts = laplace_linear_posterior(Z_ts, y)
  draws_ts = np.random.multivariate_normal(beta_hat_ts, Sigma_ts, size=laplace_samples)

  # Aggregate per-topic quantities
  for topic in range(num_topics):
      jb = draws_joint[:, topic]
      tb = draws_ts[:, topic]

      # ---- Save (percentile) ----
      store["downstream"]["percentile"]["joint"][topic].append(jb)
      store["downstream"]["percentile"]["two_stage"][topic].append(tb)

      # ---- Save (normal) ----
      m_joint, s_joint = beta_hat_joint[topic], np.sqrt(Sigma_joint[topic, topic])
      m_ts,    s_ts    = beta_hat_ts[topic],    np.sqrt(Sigma_ts[topic, topic])
      store["downstream"]["normal"]["joint"][topic].append((m_joint, s_joint))
      store["downstream"]["normal"]["two_stage"][topic].append((m_ts,    s_ts))

      # ---- Point estimates ----
      dict_downstream_joint_mc[topic].append(m_joint)
      dict_downstream_two_stage_mc[topic].append(m_ts)

      # ---- Percentile CIs + coverage vs true beta ----
      ci_joint_mc = np.percentile(jb, [q_lo, q_hi])
      ci_ts_mc    = np.percentile(tb, [q_lo, q_hi])

      if ci_joint_mc[0] <= beta[topic] <= ci_joint_mc[1]:
          downstream_coverage_mc["joint"][topic] += 1
      if ci_ts_mc[0] <= beta[topic] <= ci_ts_mc[1]:
          downstream_coverage_mc["two_stage"][topic] += 1

      # ---- Normal-based CIs + coverage ----
      ci_joint_norm = np.array([m_joint - z*s_joint, m_joint + z*s_joint])
      ci_ts_norm    = np.array([m_ts    - z*s_ts,    m_ts    + z*s_ts])

      if ci_joint_norm[0] <= beta[topic] <= ci_joint_norm[1]:
          downstream_coverage_mc["joint_norm"][topic] = downstream_coverage_mc["joint_norm"].get(topic, 0) + 1
      if ci_ts_norm[0] <= beta[topic] <= ci_ts_norm[1]:
          downstream_coverage_mc["two_stage_norm"][topic] = downstream_coverage_mc["two_stage_norm"].get(topic, 0) + 1