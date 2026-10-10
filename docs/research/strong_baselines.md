# Strong baselines for the viscosity inverse

Research note for a possible arXiv comparison. It does not replace the Stage D or Stage E reports, and it does not change their numbers. The question was frozen in `docs/research/strong_baselines_protocol.json` before these test scores were computed. Protocol SHA-256: `055aecf6ab7c3611c78652c8ddf727fe591158f51c670cb22f0556fddc129a86`.

On the Stage A harder Burgers pilot, with the Stage D sensors32_bursts mask, does the Fourier neural operator still have lower hard_ood mean relative viscosity error than a weak-form identifier and a solver-in-the-loop fit? A negative answer is an acceptable outcome.

## Estimators

Weak form: Burgers is known. Nu is the only unknown. Each test function is separable, phi(x, t) = psi(x) eta(t), with psi and eta equal to (1 - xi^2)_+^degree. Derivatives fall on phi. The field is never differentiated. Integrals use the periodic trapezoidal rule on the sensor nodes and the trapezoidal rule in time, with half-weights on the first and last frame of each burst. Bursts are pooled into one normal equation. There is no interpolation of u. Solver, sensor lift: The initial condition is the real-FFT zero-pad lift of the 32 (or fewer) sensors at frame 0 onto the N=1024 grid, the same lift_sensors map as Stage D. Modes above the sensor Nyquist are zero. This is interpolation, and it is not a Burgers state. The fit then matches the true spectral solver to the sensors on the later bursts. It does not read off-sensor samples. Solver, oracle initial condition: The initial condition is the stored full-grid field at frame 0. The fit matches the true spectral solver to the same later-burst sensors. This ceiling reads the initial condition off the sensors. It is not a sparse-sensor estimator.

## Primary mask

Mask `sensors32_bursts`: 32 equispaced sensors, four bursts of five frames at 0, 24, 48, and 72, `Δt = 0.01`. Failure rule: `ν̂ <= 0` or relative error `> 0.5`. `hard_ood` is `ν <= 0.027028120493367818`. The weak form and both solver fits are deterministic, so they have no seed standard deviation. Fourier neural operator cells are the mean ± sample standard deviation (`ddof = 1`) across seeds 0 through 4, recomputed here from the committed Stage D per-instance `lambda0_nu_hat` values at `λ = 0`. Those checkpoints were not retrained.

| estimator | slice | n | mean rel | median rel | max rel | failures | correlation |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `closed_form_ls` | `full_test` | 128 | 0.1171354466174288 | 0.04767123071353264 | 0.9497873814639219 | 9 | 0.9975830065148409 |
| `closed_form_ls` | `hard_ood` | 30 | 0.3494591802034973 | 0.165338145617806 | 0.9497873814639219 | 9 | 0.9575721159355071 |
| `closed_form_ls` | `complement` | 98 | 0.04601593633597921 | 0.04141721701903255 | 0.1965295204205419 | 0 | 0.9974501288808931 |
| `closed_form_ls` | `below_stage2_floor` | 20 | 0.4696629563210831 | 0.4955294819068466 | 0.9497873814639219 | 9 | 0.8254909890240939 |
| `weak_form` | `full_test` | 128 | 0.04766946858758207 | 0.03559816234062026 | 0.4946934593594456 | 0 | 0.9998576604094492 |
| `weak_form` | `hard_ood` | 30 | 0.08978429715170869 | 0.0523881971352341 | 0.4946934593594456 | 0 | 0.9927920034508407 |
| `weak_form` | `complement` | 98 | 0.03477717412917595 | 0.03393968269132926 | 0.05839517033138476 | 0 | 0.9999256146324119 |
| `weak_form` | `below_stage2_floor` | 20 | 0.113190760815217 | 0.07558408219388914 | 0.4946934593594456 | 0 | 0.9682275911367207 |
| `solver_sensor_ic` | `full_test` | 128 | 0.006812584961091193 | 0.003787429392221776 | 0.05513233640883729 | 0 | 0.9998911689429839 |
| `solver_sensor_ic` | `hard_ood` | 30 | 0.01379664899082717 | 0.009382715773928893 | 0.05513233640883729 | 0 | 0.9995286499972879 |
| `solver_sensor_ic` | `complement` | 98 | 0.004674606176478139 | 0.00292806808762891 | 0.01979253809315363 | 0 | 0.999785058860629 |
| `solver_sensor_ic` | `below_stage2_floor` | 20 | 0.01705426664600216 | 0.01072375329931772 | 0.05513233640883729 | 0 | 0.998765204912192 |
| `solver_oracle_ic` | `full_test` | 128 | 0.0009912558176923577 | 0.0005049933410908261 | 0.006993037410636617 | 0 | 0.9999992537672253 |
| `solver_oracle_ic` | `hard_ood` | 30 | 0.002512684392086404 | 0.002284856678533339 | 0.006993037410636617 | 0 | 0.9999852139864298 |
| `solver_oracle_ic` | `complement` | 98 | 0.000525512376551323 | 0.0003871188920198464 | 0.002332984836988702 | 0 | 0.9999988558136077 |
| `solver_oracle_ic` | `below_stage2_floor` | 20 | 0.003077124425052049 | 0.002618426296684286 | 0.006993037410636617 | 0 | 0.9999622930244444 |
| `fno_data_only` | `full_test` | 128 | 0.02186633090589915 ± 0.008819091358584748 | 0.01396562388280935 ± 0.00768733741208179 | 0.2922084272467235 ± 0.05417764942974335 | 0 ± 0 | 0.9997624518437667 ± 8.259595625306263e-05 |
| `fno_data_only` | `hard_ood` | 30 | 0.04948720846902754 ± 0.0144232271670265 | 0.0315188393346046 ± 0.01560588973425255 | 0.2922084272467235 ± 0.05417764942974335 | 0 ± 0 | 0.9962642672136852 ± 0.001268293900285726 |
| `fno_data_only` | `complement` | 98 | 0.01341096022330882 ± 0.007957033491083166 | 0.01159597558146607 ± 0.007554098931445906 | 0.03755442593267517 ± 0.01380019392435522 | 0 ± 0 | 0.9996318531207325 ± 0.0001507529185686737 |
| `fno_data_only` | `below_stage2_floor` | 20 | 0.06339223281191389 ± 0.01564338866879692 | 0.04015646987123901 ± 0.01438996841649273 | 0.2922084272467235 ± 0.05417764942974335 | 0 ± 0 | 0.9853977104393021 ± 0.004894666063483187 |
| `fno_hybrid_1e-2` | `full_test` | 128 | 0.01538195664068509 ± 0.002792467828389125 | 0.009441363956674911 ± 0.001693234421261081 | 0.1865528182798242 ± 0.05362251197593289 | 0 ± 0 | 0.9998097324167515 ± 0.0001171266310466537 |
| `fno_hybrid_1e-2` | `hard_ood` | 30 | 0.03368266375533967 ± 0.003473424738176777 | 0.01908100743845449 ± 0.002576182136905238 | 0.1865528182798242 ± 0.05362251197593289 | 0 ± 0 | 0.9979143182207665 ± 0.000680201109281932 |
| `fno_hybrid_1e-2` | `complement` | 98 | 0.009779699360688785 ± 0.002898987551908216 | 0.008203322495009369 ± 0.001797853008280559 | 0.03912937883190184 ± 0.02018618618223026 | 0 ± 0 | 0.999690804021221 ± 0.0002239885243865189 |
| `fno_hybrid_1e-2` | `below_stage2_floor` | 20 | 0.04426848215926132 ± 0.004262785004741288 | 0.02967669765325361 ± 0.006147605167564968 | 0.1865528182798242 ± 0.05362251197593289 | 0 ± 0 | 0.9933189478056381 ± 0.0009346345540773512 |

## Paired comparison

Delta is the FNO relative error minus the baseline relative error on the same instance. A negative delta is a lower error for the FNO. Counts are instances, then the mean ± sample standard deviation across seeds.

### `hard_ood`

| FNO arm | baseline | baseline mean rel | FNO mean rel | FNO − baseline | FNO lower | FNO higher | baseline failures | FNO failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `data_only` | `closed_form_ls` | 0.3494591802034973 | 0.04948720846902754 ± 0.0144232271670265 | -0.2999719717344699 ± 0.01442322716702648 | 29.4 ± 0.5477225575051661 | 0.6 ± 0.5477225575051662 | 9 | 0 ± 0 |
| `data_only` | `weak_form` | 0.08978429715170869 | 0.04948720846902754 ± 0.0144232271670265 | -0.04029708868268116 ± 0.0144232271670265 | 24 ± 4.527692569068709 | 6 ± 4.527692569068709 | 0 | 0 ± 0 |
| `data_only` | `solver_sensor_ic` | 0.01379664899082717 | 0.04948720846902754 ± 0.0144232271670265 | 0.03569055947820038 ± 0.0144232271670265 | 4.4 ± 2.509980079602227 | 25.6 ± 2.509980079602227 | 0 | 0 ± 0 |
| `data_only` | `solver_oracle_ic` | 0.002512684392086404 | 0.04948720846902754 ± 0.0144232271670265 | 0.04697452407694114 ± 0.0144232271670265 | 1.6 ± 0.5477225575051661 | 28.4 ± 0.5477225575051661 | 0 | 0 ± 0 |
| `hybrid_1e-2` | `closed_form_ls` | 0.3494591802034973 | 0.03368266375533967 ± 0.003473424738176777 | -0.3157765164481578 ± 0.003473424738176789 | 29.2 ± 0.4472135954999579 | 0.8 ± 0.447213595499958 | 9 | 0 ± 0 |
| `hybrid_1e-2` | `weak_form` | 0.08978429715170869 | 0.03368266375533967 ± 0.003473424738176777 | -0.05610163339636903 ± 0.003473424738176775 | 26.2 ± 1.483239697419132 | 3.8 ± 1.483239697419133 | 0 | 0 ± 0 |
| `hybrid_1e-2` | `solver_sensor_ic` | 0.01379664899082717 | 0.03368266375533967 ± 0.003473424738176777 | 0.01988601476451249 ± 0.003473424738176772 | 7.6 ± 0.5477225575051662 | 22.4 ± 0.5477225575051661 | 0 | 0 ± 0 |
| `hybrid_1e-2` | `solver_oracle_ic` | 0.002512684392086404 | 0.03368266375533967 ± 0.003473424738176777 | 0.03116997936325326 ± 0.003473424738176776 | 2.8 ± 0.8366600265340756 | 27.2 ± 0.8366600265340756 | 0 | 0 ± 0 |

### `complement`

| FNO arm | baseline | baseline mean rel | FNO mean rel | FNO − baseline | FNO lower | FNO higher | baseline failures | FNO failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `data_only` | `closed_form_ls` | 0.04601593633597921 | 0.01341096022330882 ± 0.007957033491083166 | -0.03260497611267039 ± 0.007957033491083168 | 85.8 ± 12.37739875741264 | 12.2 ± 12.37739875741264 | 0 | 0 ± 0 |
| `data_only` | `weak_form` | 0.03477717412917595 | 0.01341096022330882 ± 0.007957033491083166 | -0.02136621390586713 ± 0.007957033491083164 | 92.59999999999999 ± 8.763560920082659 | 5.4 ± 8.763560920082659 | 0 | 0 ± 0 |
| `data_only` | `solver_sensor_ic` | 0.004674606176478139 | 0.01341096022330882 ± 0.007957033491083166 | 0.008736354046830685 ± 0.007957033491083168 | 21 ± 10.90871211463571 | 77 ± 10.90871211463571 | 0 | 0 ± 0 |
| `data_only` | `solver_oracle_ic` | 0.000525512376551323 | 0.01341096022330882 ± 0.007957033491083166 | 0.0128854478467575 ± 0.007957033491083166 | 3.4 ± 2.408318915758459 | 94.59999999999999 ± 2.408318915758459 | 0 | 0 ± 0 |
| `hybrid_1e-2` | `closed_form_ls` | 0.04601593633597921 | 0.009779699360688785 ± 0.002898987551908216 | -0.03623623697529043 ± 0.002898987551908213 | 88 ± 4.358898943540674 | 10 ± 4.358898943540674 | 0 | 0 ± 0 |
| `hybrid_1e-2` | `weak_form` | 0.03477717412917595 | 0.009779699360688785 ± 0.002898987551908216 | -0.02499747476848716 ± 0.002898987551908212 | 96 ± 2.23606797749979 | 2 ± 2.23606797749979 | 0 | 0 ± 0 |
| `hybrid_1e-2` | `solver_sensor_ic` | 0.004674606176478139 | 0.009779699360688785 ± 0.002898987551908216 | 0.005105093184210647 ± 0.002898987551908215 | 27.4 ± 7.797435475847171 | 70.59999999999999 ± 7.797435475847171 | 0 | 0 ± 0 |
| `hybrid_1e-2` | `solver_oracle_ic` | 0.000525512376551323 | 0.009779699360688785 ± 0.002898987551908216 | 0.00925418698413746 ± 0.002898987551908214 | 4 ± 1.414213562373095 | 94 ± 1.414213562373095 | 0 | 0 ± 0 |

### `below_stage2_floor`

| FNO arm | baseline | baseline mean rel | FNO mean rel | FNO − baseline | FNO lower | FNO higher | baseline failures | FNO failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `data_only` | `closed_form_ls` | 0.4696629563210831 | 0.06339223281191389 ± 0.01564338866879692 | -0.4062707235091693 ± 0.0156433886687969 | 19.4 ± 0.5477225575051661 | 0.6 ± 0.5477225575051662 | 9 | 0 ± 0 |
| `data_only` | `weak_form` | 0.113190760815217 | 0.06339223281191389 ± 0.01564338866879692 | -0.04979852800330316 ± 0.01564338866879693 | 15.8 ± 2.16794833886788 | 4.2 ± 2.16794833886788 | 0 | 0 ± 0 |
| `data_only` | `solver_sensor_ic` | 0.01705426664600216 | 0.06339223281191389 ± 0.01564338866879692 | 0.04633796616591172 ± 0.01564338866879692 | 2 ± 1 | 18 ± 1 | 0 | 0 ± 0 |
| `data_only` | `solver_oracle_ic` | 0.003077124425052049 | 0.06339223281191389 ± 0.01564338866879692 | 0.06031510838686183 ± 0.01564338866879692 | 0.6 ± 0.5477225575051662 | 19.4 ± 0.5477225575051661 | 0 | 0 ± 0 |
| `hybrid_1e-2` | `closed_form_ls` | 0.4696629563210831 | 0.04426848215926132 ± 0.004262785004741288 | -0.4253944741618218 ± 0.004262785004741297 | 19.2 ± 0.4472135954999579 | 0.8 ± 0.447213595499958 | 9 | 0 ± 0 |
| `hybrid_1e-2` | `weak_form` | 0.113190760815217 | 0.04426848215926132 ± 0.004262785004741288 | -0.06892227865595572 ± 0.004262785004741293 | 16.6 ± 1.140175425099138 | 3.4 ± 1.140175425099138 | 0 | 0 ± 0 |
| `hybrid_1e-2` | `solver_sensor_ic` | 0.01705426664600216 | 0.04426848215926132 ± 0.004262785004741288 | 0.02721421551325916 ± 0.004262785004741288 | 4.4 ± 1.140175425099138 | 15.6 ± 1.140175425099138 | 0 | 0 ± 0 |
| `hybrid_1e-2` | `solver_oracle_ic` | 0.003077124425052049 | 0.04426848215926132 ± 0.004262785004741288 | 0.04119135773420927 ± 0.00426278500474129 | 1.2 ± 0.8366600265340756 | 18.8 ± 0.8366600265340756 | 0 | 0 ± 0 |

### `full_test`

| FNO arm | baseline | baseline mean rel | FNO mean rel | FNO − baseline | FNO lower | FNO higher | baseline failures | FNO failures |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `data_only` | `closed_form_ls` | 0.1171354466174288 | 0.02186633090589915 ± 0.008819091358584748 | -0.09526911571152966 ± 0.008819091358584753 | 115.2 ± 12.65701386583739 | 12.8 ± 12.65701386583739 | 9 | 0 ± 0 |
| `data_only` | `weak_form` | 0.04766946858758207 | 0.02186633090589915 ± 0.008819091358584748 | -0.02580313768168292 ± 0.00881909135858475 | 116.6 ± 12.85690475970014 | 11.4 ± 12.85690475970014 | 0 | 0 ± 0 |
| `data_only` | `solver_sensor_ic` | 0.006812584961091193 | 0.02186633090589915 ± 0.008819091358584748 | 0.01505374594480796 ± 0.008819091358584752 | 25.4 ± 12.30040649734796 | 102.6 ± 12.30040649734796 | 0 | 0 ± 0 |
| `data_only` | `solver_oracle_ic` | 0.0009912558176923577 | 0.02186633090589915 ± 0.008819091358584748 | 0.02087507508820679 ± 0.00881909135858475 | 5 ± 2.645751311064591 | 123 ± 2.645751311064591 | 0 | 0 ± 0 |
| `hybrid_1e-2` | `closed_form_ls` | 0.1171354466174288 | 0.01538195664068509 ± 0.002792467828389125 | -0.1017534899767437 ± 0.002792467828389127 | 117.2 ± 4.54972526643093 | 10.8 ± 4.54972526643093 | 9 | 0 ± 0 |
| `hybrid_1e-2` | `weak_form` | 0.04766946858758207 | 0.01538195664068509 ± 0.002792467828389125 | -0.03228751194689698 ± 0.002792467828389123 | 122.2 ± 3.563705936241092 | 5.8 ± 3.563705936241092 | 0 | 0 ± 0 |
| `hybrid_1e-2` | `solver_sensor_ic` | 0.006812584961091193 | 0.01538195664068509 ± 0.002792467828389125 | 0.008569371679593891 ± 0.002792467828389123 | 35 ± 8.031189202104505 | 93 ± 8.031189202104505 | 0 | 0 ± 0 |
| `hybrid_1e-2` | `solver_oracle_ic` | 0.0009912558176923577 | 0.01538195664068509 ± 0.002792467828389125 | 0.01439070082299273 ± 0.002792467828389122 | 6.8 ± 1.643167672515498 | 121.2 ± 1.643167672515498 | 0 | 0 ± 0 |

## Closed-form failures

Every test instance on which the recomputed `sensors32_bursts` least squares fails the 0.5 rule. Fourier neural operator cells are the mean relative error across the five seeds.

| instance | ν | closed form | weak form | sensor-lift solver | oracle-IC solver | data-only mean | hybrid mean |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 530 | 0.007820793287511035 | 0.9497873814639219 | 0.2076171907457948 | 0.05513233640883729 | 0.004422603080968496 | 0.09770858650905481 | 0.0538044252087569 |
| 743 | 0.0083563697313075 | 0.9365582569236526 | 0.09265776592398609 | 0.01009931681114042 | 0.004214905448180611 | 0.06505579410348099 | 0.03424670884814036 |
| 338 | 0.007689406142113342 | 0.8945644358983865 | 0.4946934593594456 | 0.02224575746838219 | 0.004843106502425292 | 0.04078906478229074 | 0.08965661708744999 |
| 619 | 0.007207675915965656 | 0.8600432285811773 | 0.1573559403167692 | 0.03255637346927215 | 0.003657005707938362 | 0.1282806774831965 | 0.1005324820410316 |
| 655 | 0.008799276647784498 | 0.8264779460832664 | 0.1141690306832752 | 0.01603497155871597 | 0.006993037410636617 | 0.04089601270207875 | 0.03177147393599544 |
| 185 | 0.006878265076856729 | 0.7136988908141626 | 0.05969675594779693 | 0.01650739330165744 | 0.002208779390371175 | 0.06915305095928528 | 0.1567641061820608 |
| 375 | 0.01137555912031582 | 0.5728204481644777 | 0.1258978425095593 | 0.0008147407600935731 | 0.0008147407600935731 | 0.03293592199273841 | 0.05055679619615903 |
| 391 | 0.01014155236080125 | 0.5328612255755112 | 0.2164759110694814 | 0.03872318041951341 | 0.002354806379648348 | 0.08464657171389177 | 0.02251237917409037 |
| 83 | 0.01183539914950419 | 0.5240345713792666 | 0.1131817634590044 | 0.0105740089468006 | 0.002870823626223841 | 0.03657841739264865 | 0.02812918820005406 |

## Does the Stage D advantage survive?

On hard_ood, the published comparison is the FNO against closed-form residual least squares (0.3494591802034973, 9 failures). The data-only FNO mean relative error (0.04948720846902754) is lower error than the weak form (0.08978429715170869). The data-only FNO mean relative error (0.04948720846902754) is higher error than the sensor-lift solver (0.01379664899082717). The data-only FNO mean relative error (0.04948720846902754) is higher error than the oracle-IC solver (0.002512684392086404). The hybrid 1e-2 FNO mean relative error (0.03368266375533967) is lower error than the weak form (0.08978429715170869). The hybrid 1e-2 FNO mean relative error (0.03368266375533967) is higher error than the sensor-lift solver (0.01379664899082717). The hybrid 1e-2 FNO mean relative error (0.03368266375533967) is higher error than the oracle-IC solver (0.002512684392086404). The advantage over the closed form does not survive as a claim that the operator beats a weak-form identifier and a solver-in-the-loop fit. At least one of those baselines has hard_ood mean relative error at or below an FNO arm. The oracle initial-condition solver is a ceiling that sees the full field at t = 0. It is not evidence about a sensor-only method.

## Compute

| estimator | seconds per instance | evaluations per instance |
| --- | ---: | ---: |
| `weak_form` | 0.001283918539002116 | null |
| `solver_sensor_ic` | 18.72235205688281 | 17.84375 |
| `solver_oracle_ic` | 16.50434028462499 | 17.84375 |
| `closed_form_ls` | 0 | null |
| `fno_stage_d_grid` | 1.847308909000276 | 191 |

Solver times are wall-clock seconds per test instance for the full coarse-grid plus golden-section search, including every forward solve, on this CPU with the worker count in the protocol. The FNO time is one instance of the same 191-point rollout the Stage D inverse evaluates. The network is the Stage D architecture with random initial weights, because the published checkpoints are not in the repository. The arithmetic does not depend on those weights. Closed-form time was not the object of the measurement and is stored as 0.

## Stage E noise and sparsity

Weak form is scored on the full Stage E condition list at noise seed 0. The solver-in-the-loop stress is only the sensor-lift initial condition, only hard_ood, and only the four Stage E cells where the published operator mean relative error crossed 10% or the sensor count dropped. A full solver grid was outside the CPU budget. Published Stage E aggregates in the comparison columns are read from docs/v02/stage_e_scores.json by this script. They are not a new operator run. The one-burst mask has no solver target and is not in the solver list. Noise is the Stage E full-field Gaussian model. After the noisy field is built, each estimator reads only its mask.

### `weak_form`

| condition | slice | weak form mean rel | weak form failures | closed form | data-only FNO | hybrid FNO |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `noise0_sensors32_bursts4` | `hard_ood` | 0.08978429715170869 | 0 | 0.3494591802034973 | 0.05103685408807077 ± 0.01549021657598742 | 0.03518458454568572 ± 0.004937207443766122 |
| `noise0_sensors32_bursts4` | `complement` | 0.03477717412917595 | 0 | 0.04601593633597921 | 0.01598345046166558 ± 0.008675434606593107 | 0.01247907452171463 ± 0.003529571100832042 |
| `noise0p001_sensors32_bursts4` | `hard_ood` | 0.09204146398622395 | 0 | 0.3496818082964408 | 0.05073194935005972 ± 0.01525108477284248 | 0.03574307451745953 ± 0.007815244996617425 |
| `noise0p001_sensors32_bursts4` | `complement` | 0.03494306886339641 | 0 | 0.04591074681525995 | 0.01613719658622993 ± 0.008593485899524973 | 0.01228949405657774 ± 0.003649378026939434 |
| `noise0p005_sensors32_bursts4` | `hard_ood` | 0.1144756021661514 | 0 | 0.3499367023840383 | 0.06140552304742215 ± 0.01772414554296476 | 0.04487067715293543 ± 0.01088399989874261 |
| `noise0p005_sensors32_bursts4` | `complement` | 0.03729620244578108 | 0 | 0.04429274450751018 | 0.01847672716239298 ± 0.007645951597203881 | 0.01505874986144249 ± 0.002739850054896627 |
| `noise0p01_sensors32_bursts4` | `hard_ood` | 0.1477967844392093 | 0 | 0.3488269447280259 | 0.08760886680179443 ± 0.02163109879512942 | 0.0692028550057078 ± 0.02303975913153131 |
| `noise0p01_sensors32_bursts4` | `complement` | 0.0490603371782533 | 0 | 0.04021816558339093 | 0.02462039560574818 ± 0.007163920113609834 | 0.02182948385755366 ± 0.004287821075015747 |
| `noise0p02_sensors32_bursts4` | `hard_ood` | 0.2426317450830368 | 4 | 0.3424017303909119 | 0.1787621128661764 ± 0.06092102725507964 | 0.1831596931330138 ± 0.04463984211913347 |
| `noise0p02_sensors32_bursts4` | `complement` | 0.07773499442913068 | 0 | 0.0332507777732665 | 0.04414658249540213 ± 0.009424425715480956 | 0.04179259616454005 ± 0.008162379728076131 |
| `noise0p05_sensors32_bursts4` | `hard_ood` | 0.5912730006403241 | 15 | 0.2989632604627805 | 0.7504081796582944 ± 0.1974892303650587 | 0.8354333660501994 ± 0.1429035982617707 |
| `noise0p05_sensors32_bursts4` | `complement` | 0.1747318038718979 | 3 | 0.09782175319232191 | 0.1326933912555423 ± 0.02880918268110285 | 0.1356247709131765 ± 0.02746389591907043 |
| `noise0_sensors16_bursts4` | `hard_ood` | 1.81489943172229 | 24 | 0.6532621372831862 | 0.1754277155678029 ± 0.01817697590235724 | 0.161153950625373 ± 0.008412564689937736 |
| `noise0_sensors16_bursts4` | `complement` | 0.3494332710564841 | 21 | 0.146734157168319 | 0.01897993106017722 ± 0.006103893511516129 | 0.01480268114579831 ± 0.002998630765872044 |
| `noise0_sensors8_bursts4` | `hard_ood` | 82.66361299910749 | 30 | 2.360147613179349 | 2.068656843257982 ± 0.0600460567376145 | 2.029706492502906 ± 0.0180751263191307 |
| `noise0_sensors8_bursts4` | `complement` | 41.32055285754434 | 94 | 0.9715662732457563 | 0.2064280084247325 ± 0.01412146227574119 | 0.1837984557289158 ± 0.01510804017528541 |
| `noise0_sensors32_bursts2` | `hard_ood` | 0.107644973588698 | 1 | 0.3174311399897838 | 0.05095427608788657 ± 0.0101037797472209 | 0.04024333453344432 ± 0.008550449407186347 |
| `noise0_sensors32_bursts2` | `complement` | 0.03509594410177325 | 0 | 0.04704341822411279 | 0.01680217837334387 ± 0.008480182488478523 | 0.01292405738084335 ± 0.003815879217048053 |
| `noise0_sensors32_bursts1` | `hard_ood` | 0.1196291686139479 | 2 | 0.296435606932363 | inapplicable | inapplicable |
| `noise0_sensors32_bursts1` | `complement` | 0.03571734431272473 | 0 | 0.04738371726979483 | inapplicable | inapplicable |
| `noise0p01_sensors16_bursts4` | `hard_ood` | 2.178357317599922 | 26 | 0.6494661505370392 | 0.2093953056335817 ± 0.02206306676193217 | 0.1998053956154922 ± 0.01802992965010784 |
| `noise0p01_sensors16_bursts4` | `complement` | 0.3876868556717826 | 27 | 0.1446215888467161 | 0.0316561400905938 ± 0.0059130363511398 | 0.0304039330757957 ± 0.004362538057948881 |
| `noise0p01_sensors8_bursts4` | `hard_ood` | 80.00110606856032 | 30 | 2.382305605141864 | 2.080554275316925 ± 0.06151776674170905 | 2.032403181974236 ± 0.01869071562754012 |
| `noise0p01_sensors8_bursts4` | `complement` | 41.1910099646125 | 94 | 0.9657909149757007 | 0.2127352042370759 ± 0.01189234399066631 | 0.1932683374306207 ± 0.01198979136846308 |
| `noise0p05_sensors8_bursts4` | `hard_ood` | 75.00666757438663 | 29 | 2.458811541682985 | 2.664783022698864 ± 0.1693654953246559 | 2.884610902035281 ± 0.1285297935248395 |
| `noise0p05_sensors8_bursts4` | `complement` | 41.29074626949135 | 95 | 0.9499594158918943 | 0.3241518628736709 ± 0.008820827465297365 | 0.3317620539598978 ± 0.01121266587451087 |
| `noise0p01_sensors32_bursts2` | `hard_ood` | 0.2035940283302016 | 2 | 0.3201645641094714 | 0.08276392487151121 ± 0.01403829198169941 | 0.08293625090732966 ± 0.01036799394674503 |
| `noise0p01_sensors32_bursts2` | `complement` | 0.05044998598855143 | 0 | 0.04357359449835964 | 0.02438649102872201 ± 0.00685384142158546 | 0.02242501837449267 ± 0.003553857255721789 |

### `solver_sensor_ic`

| condition | hard_ood n | sensor-lift solver mean rel | failures | closed form | data-only FNO | hybrid FNO |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `noise0p02_sensors32_bursts4` | 30 | 0.01420591117118668 | 0 | 0.3424017303909119 | 0.1787621128661764 ± 0.06092102725507964 | 0.1831596931330138 ± 0.04463984211913347 |
| `noise0p05_sensors32_bursts4` | 30 | 0.01822853475529066 | 0 | 0.2989632604627805 | 0.7504081796582944 ± 0.1974892303650587 | 0.8354333660501994 ± 0.1429035982617707 |
| `noise0_sensors16_bursts4` | 30 | 0.1539690293388012 | 2 | 0.6532621372831862 | 0.1754277155678029 ± 0.01817697590235724 | 0.161153950625373 ± 0.008412564689937736 |
| `noise0_sensors8_bursts4` | 30 | 2.225297649002194 | 22 | 2.360147613179349 | 2.068656843257982 ± 0.0600460567376145 | 2.029706492502906 ± 0.0180751263191307 |

## Limitations

- One scalar viscosity, one-dimensional periodic Burgers, and the Stage A pilot. This is not a multi-equation benchmark.
- The weak form never interpolates u. With five frames in a burst the time integral is a five-node trapezoidal rule. Degree, radius, and center stride were fixed before test scoring and were not swept.
- The sensor-lift solver interpolates frame 0 only. The Stage D operator also sees frames 1 through 4. Both match the later bursts and not the opening burst.
- The oracle initial-condition solver reads the full field at t = 0. It is a ceiling, not a sensor-only method.
- The solver search is a 7-point coarse grid plus at most 10 golden-section steps, down to a bracket of width 0.0005. It is not an exhaustive 191-point grid. A missed local minimum would be a property of that search.
- Fourier neural operator errors are the committed Stage D per-instance estimates at lambda 0. Those networks were not retrained in this run, and the published checkpoints are not in the repository.
- The Fourier neural operator wall clock is one instance of the Stage D 191-point rollout with a randomly initialized network of the Stage D architecture. It measures the arithmetic, not a published checkpoint.
- Solver stress covers four Stage E cells on hard_ood only. The weak form covers the full Stage E condition list at noise seed 0.
- Stage E noise is scaled by the clean full-field standard deviation. The estimators then read the mask. They do not read the clean field.

Tables in this file were written by `python -m pinnforge.operator strong-baselines` from `docs/research/strong_baselines_scores.json`. They were not typed by hand.
