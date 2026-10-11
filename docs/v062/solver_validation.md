# v062 solver validation

This note is the human reading of [solver_validation.json](solver_validation.json). It is not a test score. It does not edit [docs/v061/](../v061/) or earlier study trees. The dataset is The Well `shear_flow` (CC BY 4).

The numpy solver uses the Dedalus IC family in `RudyMorel/the-well-rbc-sf` (`init_standard_shear`) and the periodic incompressible NS + tracer equations on the The Well shear_flow domain.

Pre-stated IC gate: each of `ux`, `uy`, tracer rel L2 `<= 0.02`.
Pre-stated window gate: start from The Well frame `160`, integrate `0.8`, require `ux`, `uy`, tracer rel L2 `<= 0.15`.

Overall: `pass`.

## Matched IC

- `n_shear` `2`
- `n_blobs` `3`
- `width` `4`
- IC rel L2 ux `0.004621310353043073`
- IC rel L2 uy `0.01440121125884977`
- IC rel L2 tracer `0.004621310353043073`
- IC rel L2 pressure `0`

## Identifiable window

- ux `0.05874591303466833`
- uy `0.1139667419119713`
- pressure `0.0853974869743909`
- tracer `0.001520735889823046`

