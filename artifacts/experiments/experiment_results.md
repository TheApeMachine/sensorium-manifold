# Experiment Results

Generated: 2026-02-01 13:46:06

## Summary

| Experiment | Scale | Success | Key Metric |
|------------|-------|---------|------------|
| timeseries | toy | ✓ | MSE: 0.0216 |
| next_token | toy | ✓ | Acc: 28.51% |
| image_gen | toy | ✓ | MSE: 0.0961 |
| audio_gen | toy | ✓ | MSE: 0.0316 |
| text_diffusion | toy | ✓ | Acc: 14.37% |

## Detailed Results

### timeseries (toy)

- **mse**: 0.02163184352684766
- **mae**: 0.1300102099776268
- **rmse**: 0.14707767854724815
- **eval_samples**: 4

### next_token (toy)

- **accuracy**: 0.28514588859416445
- **perplexity**: 800.7435302734375
- **eval_tokens**: 9048
- **graph_edges**: 1673
- **chunks**: 4016

### image_gen (toy)

- **reconstruction_mse**: 0.09606941517442465
- **gen_mean**: 8.238334703492001e-05
- **gen_std**: 0.0003465304325800389
- **mean_diff**: 0.12651260714483215
- **std_diff**: 0.30287934446823783
- **num_attractors**: 278
- **eval_images**: 50

### audio_gen (toy)

- **reconstruction_mse**: 0.03157374002039433
- **spectral_distance**: 252.52994079589843
- **gen_energy**: 5.520884951693006e-05
- **num_freq_attractors**: 557
- **eval_samples**: 20

### text_diffusion (toy)

- **accuracy**: 0.14375
- **perplexity**: 222.70412216186523
- **samples_evaluated**: 20
- **graph_edges**: 3853
- **chunks**: 19855
