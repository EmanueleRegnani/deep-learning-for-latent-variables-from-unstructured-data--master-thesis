output_csv = f"../logs/document-topic_shares/mixed_GS/{doc_topic_prior}_results.csv"
write_header = not os.path.exists(output_csv)  # only write header if file doesn't exist

for enc in encoder_layer_configs:
  for dec in decoder_layer_configs:
    for w_prior in w_prior_configs:
      # ...
      # ...
      # ...

      # Plotting
      n_cols = 2
      n_rows = (num_topics + 1) // 2
      fig, axs = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
      axs = axs.flatten()

      for i in range(num_topics):
          x = reanged_df_gtm[f'Topic{i}']
          y = true_df[f'Topic{i}']

          axs[i].scatter(x, y, label='Data points', s=1, color='grey', alpha=0.2)
          coefficients = np.polyfit(x, y, 1)
          fit = np.poly1d(coefficients)
          axs[i].plot(x, fit(x), color='black', label='Linear Fit', alpha=0.8)
          axs[i].set_xlabel('Estimates', fontsize=15)
          axs[i].set_ylabel('True Value', fontsize=15)
          axs[i].set_title(f'Topic {i}', fontsize=15)
          axs[i].legend(fontsize=12)

          corr_coeff = np.corrcoef(x, y)[0, 1]
          axs[i].annotate(f'Correlation: {corr_coeff:.2f}', xy=(0.55, 0.3), xycoords='axes fraction',
                          fontsize=12, verticalalignment='top', horizontalalignment='left',
                          bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

      for j in range(num_topics, n_rows * n_cols):
          fig.delaxes(axs[j])

      plt.suptitle(f'Encoder: {enc} | Decoder: {dec} | w_prior: {w_prior}', fontsize=16, y=1.02)
      plt.subplots_adjust(top=0.75)
      plt.tight_layout()

      plot_path = f"../figs/document-topic_shares/true_vs_estimated_{doc_topic_prior}__enc_{enc}_dec_{dec}_wprior_{w_prior}.pdf"
      plt.savefig(plot_path)
      plt.close()


      # Compute correlations
      topic_corrs = [
          np.corrcoef(reanged_df_gtm[f"Topic{i}"], true_df[f"Topic{i}"])[0, 1]
          for i in range(num_topics)
      ]
      row = {
          "w_prior": w_prior, "encoder": enc, "decoder": dec, "mean_corr": np.mean(topic_corrs),
          **{f"corr_topic_{i}": topic_corrs[i] for i in range(num_topics)}
      }

      # Append to CSV
      with open(output_csv, mode='a', newline='') as f:
          writer = csv.DictWriter(f, fieldnames=row.keys())
          if write_header:
              writer.writeheader()
              write_header = False
          writer.writerow(row)
