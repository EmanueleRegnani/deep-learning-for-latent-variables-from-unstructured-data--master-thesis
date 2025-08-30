# === Grid Search Loop ===
for enc, dec, w_prior, w_pred_loss in itertools.product(encoder_layer_configs, decoder_layer_configs, w_prior_configs, w_pred_loss_configs):
    # ...
    # ...
    # ...

    # === Results CSV ===
    print("\n\n  -- Saving Results in CSV")
    results = []

    if mode in ['upstream', 'both']:
        for approach_name, upstream_dict in zip(
            ['joint', 'two_stage', 'infeasible'],
            [dict_upstream_joint, dict_upstream_two_stage, dict_upstream_infeasible]
        ):
            for topic in range(num_topics):
                estimates = upstream_dict[topic]
                results.append({
                    'model': approach_name,
                    'direction': 'upstream',
                    'topic': topic,
                    'true_value': lambda_[1, topic],
                    'mean_estimate': np.mean(estimates),
                    'std_dev_0': np.std(estimates),
                    'std_dev_1': np.std(estimates, ddof=1),
                    'std_error_0': np.std(estimates) / np.sqrt(len(estimates)),
                    'std_error_1': np.std(estimates, ddof=1) / np.sqrt(len(estimates)),
                    'containment_0': np.mean((estimates > lambda_[1, topic] - 1.96*np.std(estimates)) &
                                        (estimates < lambda_[1, topic] + 1.96*np.std(estimates))),
                    'containment_1': np.mean((estimates > lambda_[1, topic] - 1.96*np.std(estimates, ddof=1)) &
                                                    (estimates < lambda_[1, topic] + 1.96*np.std(estimates, ddof=1))),
                    'coverage': upstream_coverage[approach_name][topic] / num_iter
                })

    if mode in ['downstream', 'both']:
        for approach_name, downstream_dict in zip(
            ['joint', 'two_stage', 'infeasible'],
            [dict_downstream_joint, dict_downstream_two_stage, dict_downstream_infeasible]
        ):
            for topic in range(num_topics):
                estimates = downstream_dict[topic]
                results.append({
                    'model': approach_name,
                    'direction': 'downstream',
                    'topic': topic,
                    'true_value': beta[topic],
                    'mean_estimate': np.mean(estimates),
                    'std_dev_0': np.std(estimates),
                    'std_dev_1': np.std(estimates, ddof=1),
                    'std_error_0': np.std(estimates) / np.sqrt(len(estimates)),
                    'std_error_1': np.std(estimates, ddof=1) / np.sqrt(len(estimates)),
                    'containment_0': np.mean((estimates > beta[topic] - 1.96*np.std(estimates)) &
                                                    (estimates < beta[topic] + 1.96*np.std(estimates))),
                    'containment_1': np.mean((estimates > beta[topic] - 1.96*np.std(estimates, ddof=1)) &
                                                    (estimates < beta[topic] + 1.96*np.std(estimates, ddof=1))),
                    'coverage': downstream_coverage[approach_name][topic] / num_iter
                })

    summary_df = pd.DataFrame(results)

    csv_path = f"{prefix}/logs/upstream_and_downstream/{suffix}.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"Results saved at '{csv_path}'")


    # === Plots ===
    print("\n\n  -- Plotting Results")
    approaches = ['joint', 'two_stage', 'infeasible']
    approach_dicts_upstream = {
        'joint': dict_upstream_joint,
        'two_stage': dict_upstream_two_stage,
        'infeasible': dict_upstream_infeasible
    }
    approach_dicts_downstream = {
        'joint': dict_downstream_joint,
        'two_stage': dict_downstream_two_stage,
        'infeasible': dict_downstream_infeasible
    }

    def add_subplot(ax, x_list, y_list, true_value=None, title=None, ci_band=True, fit_line=True, corr=True, kind='scatter'):
        if kind == 'scatter':
            all_corrs = []
            for x, y in zip(x_list, y_list):
                ax.scatter(x, y, alpha=0.1, s=1, color='gray')
                if fit_line:
                    coef = np.polyfit(x, y, 1)
                    poly1d_fn = np.poly1d(coef)
                    ax.plot(np.sort(x), poly1d_fn(np.sort(x)), alpha=0.3, color='black')
                if corr:
                    corr_val = np.corrcoef(x, y)[0, 1]
                    all_corrs.append(corr_val)

            if corr and all_corrs:
                mean_corr = np.mean(all_corrs)
                ax.annotate(f'Mean Corr: {mean_corr:.2f}', xy=(0.05, 0.9), xycoords='axes fraction', fontsize=12,
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)

        elif kind == 'hist':
            sns.histplot(x_list, kde=True, stat='density', ax=ax, color='gray', alpha=0.5)
            if ci_band:
                ci_low, ci_high = np.percentile(x_list, [2.5, 97.5])
                ax.axvspan(ci_low, ci_high, color='gray', alpha=0.2)

            ax.axvline(true_value, color='black', linestyle='--')

        if title:
            ax.set_title(title)

    # Adjust the number of rows based on mode
    if mode == 'upstream' or mode == 'downstream':
        n_rows = 2
        width = 8
    else:
        n_rows = 3
        width = 12

    fig, axs = plt.subplots(n_rows, 6, figsize=(24, width))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    plot_positions = { (approach, topic): col for topic in [0, 1] for col, approach in enumerate(approaches, start=topic * len(approaches)) }

    for (approach, topic), col in plot_positions.items():
        doc_topic_list = estimated_doc_topic_list_joint if approach == 'joint' else (
            estimated_doc_topic_list_two_stage if approach == 'two_stage' else true_doc_topic_list)

        x_list = []
        y_list = []
        for i in range(num_iter):
            x_vals = doc_topic_list[i].iloc[:, topic].values
            y_vals = true_doc_topic_list[i].iloc[:, topic].values
            x_list.append(x_vals)
            y_list.append(y_vals)

        ax = axs[0, col]
        add_subplot(ax, x_list, y_list)
        if col == 0:
            ax.set_xlabel("Estimated")
            ax.set_ylabel("True")

        if mode == 'upstream':
            # Upstream histogram
            x_up_hist_list = [approach_dicts_upstream[approach][topic][i] for i in range(num_iter)]
            add_subplot(axs[1, col], x_up_hist_list, None, true_value=lambda_[1, topic], kind='hist')
        elif mode == 'downstream':
            # Downstream histogram
            x_down_hist_list = [approach_dicts_downstream[approach][topic][i] for i in range(num_iter)]
            add_subplot(axs[1, col], x_down_hist_list, None, true_value=beta[topic], kind='hist')
        else:
            # Upstream histogram
            x_up_hist_list = [approach_dicts_upstream[approach][topic][i] for i in range(num_iter)]
            add_subplot(axs[1, col], x_up_hist_list, None, true_value=lambda_[1, topic], kind='hist')
            # Downstream histogram
            x_down_hist_list = [approach_dicts_downstream[approach][topic][i] for i in range(num_iter)]
            add_subplot(axs[2, col], x_down_hist_list, None, true_value=beta[topic], kind='hist')

    # Adjust labels based on mode
    if mode == 'upstream':
        axs[1, 0].set_ylabel("Upstream", fontsize=14, labelpad=30)
    elif mode == 'downstream':
        axs[1, 0].set_ylabel("Downstream", fontsize=14, labelpad=30)
    else:
        axs[1, 0].set_ylabel("Upstream", fontsize=14, labelpad=30)
        axs[2, 0].set_ylabel("Downstream", fontsize=14, labelpad=30)

    for i, approach in enumerate(approaches):
      axs[n_rows-1, i].set_xlabel(f'{approach}', fontsize=14, labelpad=30)
      axs[n_rows-1, i+3].set_xlabel(f'{approach}', fontsize=14, labelpad=30)

    # Add group titles above Topic 0 and Topic 1 columns
    fig.text(0.31, 0.92, "Topic 0", ha='center', fontsize=18)
    fig.text(0.71, 0.92, "Topic 1", ha='center', fontsize=18)

    path = f"{prefix}/figs/upstream_and_downstream/{suffix}.png"
    plt.savefig(path, format='png', dpi=300)
    print(f"Plots saved at '{path}'")
    # plt.show()
    plt.close()