# Burgers inverse demo

Command: `pinnforge demo burgers-inverse --output runs/burgers-inverse`

Small CPU example of sparse viscosity recovery. Not a rerun of the Stage D sensors32_bursts study.

## Results

```text
estimator                n   mean abs  mean rel  max rel  failures
-----------------------  -  ---------  --------  -------  --------
closed form, 32 sensors  4  1.670e-04     0.41%    0.71%         0
FNO sensor match         4  2.556e-03     4.97%   11.85%         0
training-mean constant   4  1.052e-02    25.93%   54.20%         1
```

On this smooth sample the closed form wins (0.41% vs 4.97% mean relative error). On the hard-OOD Stage D slice the FNO mean relative error is lower than the closed form (STAGE_D_REPORT.md) and higher than the sensor-lift solver (docs/research/strong_baselines.md).

### Full precision

| estimator | n | mean abs error | mean rel error | max rel error | failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| closed form, 32 sensors | 4 | 1.66974755e-04 | 4.05968068e-03 | 7.08700857e-03 | 0 |
| FNO sensor match | 4 | 2.55609737e-03 | 4.96650095e-02 | 1.18468303e-01 | 0 |
| training-mean constant | 4 | 1.05151535e-02 | 2.59304474e-01 | 5.41994932e-01 | 1 |

Data-only FNO, width 24, modes 8, layers 2, 40 Adam epochs, seed 0. Selected epoch 40 by validation relative L2 3.69904987e-02. Wall clock 2.77327819e+00 seconds on CPU with torch threads set to 1. Viscosity grid step 1.00000000e-03.

Normal equation of the central residual on every saved frame at equispaced sensors. R = a - nu * b.

Trained data-only FNO. The first window is lifted from the sensors. Later blocks are an autoregressive rollout. The grid point with the smallest sensor mean square is the estimate. Ties keep the smaller viscosity.

Training-split mean viscosity, the same value for every test instance.

Failure rule: nu_hat <= 0 or rel_error > 0.5.

## Plots

Blue points in `viscosity_scatter.png` are the closed form. Orange points are the FNO. The gray line is equality.

`relative_error.png` is the mean relative error of the closed form, the FNO, and the training-mean constant. The axis is log10 when the bars span more than a factor of eight. The numbers on the bars are the relative errors.

`field_sensors.png` is the first test trajectory at the first and last saved frames. Black dots are the sensors on the first frame.

Per-instance rows are in `result.json`.
