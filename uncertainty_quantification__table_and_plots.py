# === Results CSV ===
print("\n\n  -- Computing CSV Results\n")
results = []

def compute_uq_stats(uq_data, num_iter, uq_samples):
    """
    Compute statistics for uncertainty quantification estimates.

    uq_data: list of lists, where each inner list contains uq_samples estimates for one iteration
    """
    if not uq_data:
        return {
            'uq_mean': np.nan, 'uq_std_dev_0': np.nan, 'uq_std_dev_1': np.nan,
            'uq_std_error_0': np.nan, 'uq_std_error_1': np.nan,
            'uq_within_iter_mean_std': np.nan, 'uq_between_iter_std': np.nan
        }

    # Flatten all UQ estimates across all iterations
    all_uq_estimates = [est for iter_estimates in uq_data for est in iter_estimates]

    # Within-iteration variability (average std dev across iterations)
    within_iter_stds = [np.std(iter_estimates, ddof=1) if len(iter_estimates) > 1 else 0
                      for iter_estimates in uq_data]
    uq_within_iter_mean_std = np.mean(within_iter_stds)
    uq_within_iter_std_of_std = np.std(within_iter_stds, ddof=1) if len(within_iter_stds) > 1 else 0

    # Between-iteration variability (std dev of iteration means)
    iter_means = [np.mean(iter_estimates) for iter_estimates in uq_data]
    uq_between_iter_std = np.std(iter_means, ddof=1) if len(iter_means) > 1 else 0

    return {
        'uq_mean': np.mean(all_uq_estimates),
        'uq_std_dev_0': np.std(all_uq_estimates),
        'uq_std_dev_1': np.std(all_uq_estimates, ddof=1),
        'uq_std_error_0': np.std(all_uq_estimates) / np.sqrt(len(all_uq_estimates)),
        'uq_std_error_1': np.std(all_uq_estimates, ddof=1) / np.sqrt(len(all_uq_estimates)),
        'uq_within_iter_mean_std': uq_within_iter_mean_std,
        'uq_within_iter_std_of_std': uq_within_iter_std_of_std,
        'uq_between_iter_std': uq_between_iter_std
    }

if mode in ['upstream', 'both']:
    for approach_name, upstream_dict in zip(
        ['joint', 'two_stage', 'infeasible'],
        [dict_upstream_joint, dict_upstream_two_stage, dict_upstream_infeasible]
    ):
        # Get corresponding UQ dictionary
        if approach_name == 'joint':
            uq_dict = store['upstream']['percentile']['joint']
        elif approach_name == 'two_stage':
            uq_dict = store['upstream']['percentile']['two_stage']
        else:  # infeasible - no UQ for this
            uq_dict = {k: [] for k in range(num_topics)}

        for topic in range(num_topics):
            # Monte Carlo estimates (across iterations)
            uq_estimates = upstream_dict[topic]

            # Monte Carlo coverage calculation
            mc_coverage = upstream_coverage[approach_name][topic] / num_iter

            # UQ coverage calculation
            uq_coverage = 0
            if approach_name != 'infeasible':
                uq_coverage = upstream_coverage_uq[approach_name][topic] / num_iter

            # UQ coverage calculation (Normal-based intervals)
            uq_coverage_norm = 0
            if approach_name != 'infeasible':
                uq_coverage_norm = upstream_coverage_uq[approach_name + "_norm"][topic] / num_iter

            # Compute UQ statistics
            uq_stats = compute_uq_stats(uq_dict[topic], num_iter, uq_samples) if approach_name != 'infeasible' else {
                'uq_mean': np.nan, 'uq_std_dev_0': np.nan, 'uq_std_dev_1': np.nan,
                'uq_std_error_0': np.nan, 'uq_std_error_1': np.nan,
                'uq_within_iter_mean_std': np.nan, 'uq_within_iter_std_of_std': np.nan,
                'uq_between_iter_std': np.nan
            }

            results.append({
                'model': approach_name,
                'direction': 'upstream',
                'topic': topic,
                'true_value': lambda_[1, topic],

                # Monte Carlo Statistics (across iterations)
                'mc_mean': np.mean(uq_estimates),
                'mc_std_dev_0': np.std(uq_estimates),
                'mc_std_dev_1': np.std(uq_estimates, ddof=1) if len(uq_estimates) > 1 else np.nan,
                'mc_std_error_0': np.std(uq_estimates) / np.sqrt(len(uq_estimates)) if len(uq_estimates) > 0 else np.nan,
                'mc_std_error_1': np.std(uq_estimates, ddof=1) / np.sqrt(len(uq_estimates)) if len(uq_estimates) > 1 else np.nan,

                # UQ Statistics (within and across iterations)
                'uq_mean': uq_stats['uq_mean'],
                'uq_std_dev_0': uq_stats['uq_std_dev_0'],
                'uq_std_dev_1': uq_stats['uq_std_dev_1'],
                'uq_std_error_0': uq_stats['uq_std_error_0'],
                'uq_std_error_1': uq_stats['uq_std_error_1'],
                'uq_within_iter_std': uq_stats['uq_within_iter_mean_std'],
                'uq_within_iter_std_of_std': uq_stats['uq_within_iter_std_of_std'],
                'uq_between_iter_std': uq_stats['uq_between_iter_std'],

                # Coverage rates
                'coverage_mc': mc_coverage,
                'coverage_uq_dropout': uq_coverage,
                'coverage_uq_norm': uq_coverage_norm,

                # Experiment parameters
                'num_iter': num_iter,
                'uq_samples': uq_samples
            })

if mode in ['downstream', 'both']:
    for approach_name, downstream_dict in zip(
        ['joint', 'two_stage', 'infeasible'],
        [dict_downstream_joint, dict_downstream_two_stage, dict_downstream_infeasible]
    ):
        # Get corresponding UQ dictionary
        if approach_name == 'joint':
            uq_dict = store['downstream']['percentile']['joint']
        elif approach_name == 'two_stage':
            uq_dict = store['downstream']['percentile']['two_stage']
        else:  # infeasible - no UQ for this
            uq_dict = {k: [] for k in range(num_topics)}

        for topic in range(num_topics):
            # Monte Carlo estimates (across iterations)
            mc_estimates = downstream_dict[topic]

            # Monte Carlo coverage calculation
            mc_coverage = downstream_coverage[approach_name][topic] / num_iter

            # UQ coverage calculation
            uq_coverage = 0
            if approach_name != 'infeasible':
                uq_coverage = downstream_coverage_uq[approach_name][topic] / num_iter

            # UQ coverage calculation (Normal-based intervals)
            uq_coverage_norm = 0
            if approach_name != 'infeasible':
                uq_coverage_norm = downstream_coverage_uq[approach_name + "_norm"][topic] / num_iter

            # Compute UQ statistics
            uq_stats =(uq_dict[topic], num_iter, uq_samples) if approach_name != 'infeasible' else {
                'uq_mean': np.nan, 'uq_std_dev_0': np.nan, 'uq_std_dev_1': np.nan,
                'uq_std_error_0': np.nan, 'uq_std_error_1': np.nan,
                'uq_within_iter_mean_std': np.nan, 'uq_within_iter_std_of_std': np.nan,
                'uq_between_iter_std': np.nan
            }

            results.append({
                'model': approach_name,
                'direction': 'downstream',
                'topic': topic,
                'true_value': beta[topic],

                # Monte Carlo Statistics (across iterations)
                'mc_mean': np.mean(uq_estimates),
                'mc_std_dev_0': np.std(uq_estimates),
                'mc_std_dev_1': np.std(uq_estimates, ddof=1) if len(uq_estimates) > 1 else np.nan,
                'mc_std_error_0': np.std(uq_estimates) / np.sqrt(len(uq_estimates)) if len(uq_estimates) > 0 else np.nan,
                'mc_std_error_1': np.std(uq_estimates, ddof=1) / np.sqrt(len(uq_estimates)) if len(uq_estimates) > 1 else np.nan,

                # UQ Statistics (within and across iterations)
                'uq_mean': uq_stats['uq_mean'],
                'uq_std_dev_0': uq_stats['uq_std_dev_0'],
                'uq_std_dev_1': uq_stats['uq_std_dev_1'],
                'uq_std_error_0': uq_stats['uq_std_error_0'],
                'uq_std_error_1': uq_stats['uq_std_error_1'],
                'uq_within_iter_std': uq_stats['uq_within_iter_mean_std'],
                'uq_within_iter_std_of_std': uq_stats['uq_within_iter_std_of_std'],
                'uq_between_iter_std': uq_stats['uq_between_iter_std'],

                # Coverage rates
                'coverage_uq': mc_coverage,
                'coverage_uq_dropout': uq_coverage,
                'coverage_uq_norm': uq_coverage_norm,

                # Experiment parameters
                'num_iter': num_iter,
                'uq_samples': uq_samples
            })

summary_df = pd.DataFrame(results)

csv_path = f"{prefix}/logs/upstream_and_downstream/UQ/{suffix}.csv"
summary_df.to_csv(csv_path, index=False)
print(f"Summary Table saved at '{csv_path}'")
summary_df




# === Plotting ===
print("\n\n  -- Plotting Results")

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

coverages = summary_df['coverage_uq_dropout']
lambda_, beta = np.array([[ 0,  0], [ 0, -0.5]]), np.array([ 0, -0.5])
alpha = 0.05            # 95% CI

def plot_ci_percentile_vertical(
    estimates_per_iter,
    true_value,
    ax,
    title="",
    alpha=0.05,
    whisker_height=0.3,
    hist_height="5%",
    hist_alpha=0.75,
    coverage_id=None
):
    q_lo, q_hi = 100 * (alpha/2), 100 * (1 - alpha/2)

    ax.axvline(true_value, color='black', linestyle='--', label='True value', linewidth=1)

    offset = 0.39
    x_min, x_max = true_value - offset, true_value + offset
    ax.set_xlim(x_min, x_max)

    n_iter = 1000
    mean_per_iter = []

    for i, samples in enumerate(estimates_per_iter):
        samples = np.asarray(samples).ravel()
        lower, upper = np.percentile(samples, [q_lo, q_hi])
        mean_est = np.mean(samples)
        mean_per_iter.append(mean_est)

        covers_true = lower <= true_value <= upper
        color = 'black' if covers_true else 'red'

        y = i
        ax.plot([lower, upper], [y, y], color=color, linewidth=0.4)

    # --- 95% percentile interval of the means as gray background band ---
    if len(mean_per_iter) >= 2:
        m_lo, m_hi = np.percentile(mean_per_iter, [q_lo, q_hi])
        ax.axvspan(m_lo, m_hi, color="0.85", alpha=0.8, zorder=0, ec="none")

    # --- Top marginal histogram of per-iteration means ---
    ax_bottom = None
    if len(mean_per_iter) >= 1:
        divider = make_axes_locatable(ax)
        ax_bottom = divider.append_axes("top", size=hist_height, pad=0.1, sharex=ax)
        ax_bottom.hist(mean_per_iter, bins="auto", orientation="vertical",
                       color="0.6", alpha=hist_alpha, edgecolor="0.4")
        ax_bottom.set_yticks([])
        ax_bottom.tick_params(axis='x', labelbottom=False)
        for spine in ("top", "right", "left"):
            ax_bottom.spines[spine].set_visible(False)

    # --- Set y-axis padding to match iterations exactly ---
    ax.margins(y=0.0)
    ax.set_ylim(-whisker_height, (n_iter - 1) + whisker_height)
    ax.set_yticks(np.arange(0, n_iter+1, 100))
    if coverage_id == 1:
        ax.set_ylabel("Iterations")
    ax.set_xlabel("Values")

    # --- Title above the histogram ---
    if title:
        if ax_bottom is not None:
            ax_bottom.set_title(title, pad=6)  # title above the histogram
        else:
            ax.set_title(title)  # fallback if no histogram

    ax.annotate(f"Coverage: {round(coverages[coverage_id]*100)}%",
                xy=(0.5, 0.99), xycoords='axes fraction', fontsize=10,
                verticalalignment='top', horizontalalignment='center',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.75))

def _label(task, model, topic):
    return f"{model.replace('_','-').capitalize()} - Topic {str(topic).capitalize()}"

fig, axes = plt.subplots(1, 6, figsize=(16, 25), sharey=True)
tasks = ["upstream", "downstream"]
models = ["joint", "two_stage"]
coverage_ids = {
    0: 1,
    1: 3,
    2: 6,
    3: 8,
    4: 7,
    5: 9
}

c = 0
for task in tasks:
    for topic in [0, 1]:
        if task == 'upstream' and topic == 0:
          continue
        for model in models:
            true_value = lambda_[1, topic] if task == "upstream" else beta[topic]
            ax = axes[c]
            estimates = store[task]["percentile"][model][topic]
            plot_ci_percentile_vertical(
                estimates,
                true_value,
                ax=ax,
                title=_label(task, model, topic),
                alpha=alpha,
                coverage_id=coverage_ids[c]
            )
            c += 1

a, b, c, d = 14, 37, 7, 0
fig.suptitle(f"  {'-'*a}    Upstream    {'-'*a}{' '*c}{'-'*b}    Downstream    {'-'*b}", y = 0.9, fontsize=16)
plt.subplots_adjust(wspace=0.1)
axes[0].legend(loc='center right')

path = f"{prefix}/figs/upstream_and_downstream/UQ/{suffix}__PERCENTILE.png"
plt.savefig(path, format='png', dpi=300)
plt.show()
