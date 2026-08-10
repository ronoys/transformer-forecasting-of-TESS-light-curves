# Forecasting evaluation

Source: `test_predictions.npz` — 400 test windows, 32-step horizon, 3 models.
All errors in ppm of relative flux (flux sits at ~1.0, so 1e-6 = 1 ppm).

## Window labels

`transit_depth` is `median(window) - min(window)`, which is positive for pure noise — on this set it flags 398/400 windows. Windows are instead labelled by dip SNR (depth / robust window noise):

- transit (SNR >= 3.0): **41**
- quiet (SNR <= 1.5): **58**
- ambiguous (excluded from binary comparisons): **301**
- SNR range: 0.00 to 7.43

## Headline

| model | MAE | RMSE | P90 AE | Bias | R² | shape r | skill vs persistence |
|---|---|---|---|---|---|---|---|
| persistence | 6588.6 | 10416.9 | 18953.4 | +206.7 | -0.137 | n/a | — |
| lstm | 5421.8 | 8740.4 | 15406.1 | -443.7 | 0.199 | 0.007 | +16.1% |
| transformer | 6589.6 | 10416.5 | 18799.6 | +117.8 | -0.137 | 0.004 | +0.0% |

`shape r` is the mean per-window correlation between forecast and truth — whether the model tracks the shape of the next segment at all, which the error columns do not measure. It is `n/a` for persistence by construction: a flat forecast has no variance to correlate.

## Transit vs quiet windows

| model | MAE (transit) | RMSE (transit) | MAE (quiet) | dip depth recovered | dip floor MAE |
|---|---|---|---|---|---|
| persistence | 6139.4 | 9664.0 | 6913.7 | 0.0% | 13799.0 |
| lstm | 5406.9 | 8649.0 | 5729.1 | 1.5% | 12550.0 |
| transformer | 6138.5 | 9655.1 | 6914.5 | 2.7% | 13280.2 |

## Transit detection from forecast residuals

Score is `max(pred - true)` over the window: the largest amount the truth fell below the forecast. Positives are transit windows, negatives are quiet windows; ambiguous windows are excluded.

| model | ROC AUC | avg precision |
|---|---|---|
| persistence | 0.623 | 0.518 |
| lstm | 0.663 | 0.552 |
| transformer | 0.622 | 0.517 |

## Useful forecast horizon

- **lstm** beats persistence at 32/32 steps (best +20.4% at step 24, worst +12.8%); step 1 MAE 4775.9 ppm, step 32 MAE 5236.3 ppm
- **transformer** beats persistence at 16/32 steps (best +0.4% at step 8, worst -0.7%); step 1 MAE 5745.1 ppm, step 32 MAE 6355.4 ppm

---

Regenerate with `python evaluate.py --preds test_predictions.npz --outdir outputs --theme light`.
