# Changelog

All notable changes to this project will be documented in this file.

## [0.18.1] - 2026-04-23

### Bug Fixes

- Low_rank transform should fix the mean (Adrian Seyboldt)

- Correctly handle mean of low_rank transform (Adrian Seyboldt)

- Add proptest regression (Adrian Seyboldt)


### Features

- Add wasm32-unknown-unknown support (habakan)


### Miscellaneous Tasks

- Delete useless example (Adrian Seyboldt)


### Testing

- Add correlated normal test (Adrian Seyboldt)

- Verify mvnormal can be fit exactly (Adrian Seyboldt)

- Unify testing density (Adrian Seyboldt)


## [0.18.0] - 2026-04-20

### Bug Fixes

- Gate cargo test/example targets behind zarr feature (#61) (Eliot Carlson)

- Left over warnings (Adrian Seyboldt)

- Reinitialize total momentum on divergence in mclmc (Adrian Seyboldt)

- Incorrect momentum refresh in mclmc (Adrian Seyboldt)

- Mclmc fix various small issues (Adrian Seyboldt)

- Change mclmc defaults (Adrian Seyboldt)

- Zarr string did not need to be optional (Adrian Seyboldt)


### Documentation

- Add small docstrings to all modules (Adrian Seyboldt)

- Add experimental markers for mclmc (Adrian Seyboldt)


### Features

- Switch random number generator to ChaCha8 (Adrian Seyboldt)

- Speed up mass matrix transformations (Adrian Seyboldt)

- Implement target integration time option in nuts (Adrian Seyboldt)

- Implement trajectory relative to normal dist in transformed hmc (Adrian Seyboldt)

- Grow adaptation windows in second phase (Adrian Seyboldt)

- Add extra_doublings parameter (Adrian Seyboldt)

- Experimental microcanonical update (Adrian Seyboldt)

- Implement MCLMC (Adrian Seyboldt)

- Add mclmc sample weight (Adrian Seyboldt)

- Add options to control collection frequency in mclmc (Adrian Seyboldt)

- Implement dynamic step size in mclmc (Adrian Seyboldt)

- Implement deserialize for Settings (Adrian Seyboldt)

- Add missing mclmc sampler settings (Adrian Seyboldt)

- Store sampler and adaptation in zarr attr (Adrian Seyboldt)

- Better nuts-derive with support for strings (Adrian Seyboldt)

- Store divergence message (Adrian Seyboldt)

- Store sample stats that appear occisionally with different dims (Adrian Seyboldt)

- Store transformation updates sparsely (Adrian Seyboldt)

- Add max_energy_error diagnostic (Adrian Seyboldt)

- Use standard hmc early in mclmc (Adrian Seyboldt)


### Miscellaneous Tasks

- Remove duplicate runs (#54) (Will Dean)


### Refactor

- Only use transformed hamiltonian (Adrian Seyboldt)

- Restructure directories (Adrian Seyboldt)

- Move some sampler stats to proper places (Adrian Seyboldt)


### Styling

- Clippy fixes (Adrian Seyboldt)


## [0.17.4] - 2026-02-18

### Bug Fixes

- Allow empty arrays in zarr (Adrian Seyboldt)


### Miscellaneous Tasks

- Bump rand to 0.10 (Adrian Seyboldt)

- Bump zarrs (Adrian Seyboldt)


## [0.17.3] - 2025-11-27

### Features

- Compress zarr traces with zstd (Adrian Seyboldt)


## [0.17.2] - 2025-11-27

### Bug Fixes

- Handle backpressure in async zarr storage (Adrian Seyboldt)


### Miscellaneous Tasks

- Update pulp dependency (Adrian Seyboldt)


## [0.17.1] - 2025-11-13

### Bug Fixes

- Store step size info in transform_adapt_strategy (Adrian Seyboldt)

- Mindepth when check_turning=True was misbehaving (Adrian Seyboldt)


### Features

- Support datetime coordinates (Adrian Seyboldt)


### Miscellaneous Tasks

- Update dependencies (Adrian Seyboldt)


## [0.17.0] - 2025-10-08

### Bug Fixes

- Disable unused faer features (Adrian Seyboldt)

- Step size jitter (Adrian Seyboldt)

- Restore dual-average step size adapt as default (Adrian Seyboldt)

- Fix code in README USAGE section (aaelony)

- Remove extra block in async zarr (Adrian Seyboldt)

- Correctly specify dims of some sample stats (Adrian Seyboldt)

- Missing mut in nuts-derive (Adrian Seyboldt)


### Features

- Allow sampling with fixed step size (Adrian Seyboldt)

- Add step size jitter (Adrian Seyboldt)

- Add mindepth option for nuts (Adrian Seyboldt)

- Enable step size jitter by default (Adrian Seyboldt)

- Implement step size adaptation with adam (Adrian Seyboldt)

- Generalize sample and stats storage (Adrian Seyboldt)

- Add csv file storage backend (Adrian Seyboldt)

- Implement async zarr storage (Adrian Seyboldt)

- Add rng to Model.math() (Adrian Seyboldt)

- Bring back Storage.inspect (Adrian Seyboldt)

- Implement arrow storage (Adrian Seyboldt)


### Miscellaneous Tasks

- Update dependencies (Adrian Seyboldt)

- Correctly specify dependencies in workspace (Adrian Seyboldt)


### Performance

- Shortcut for empty posteriors (Adrian Seyboldt)


### Refactor

- Clean up some left overs from new storage (Adrian Seyboldt)


### Styling

- Some formatting changes (Adrian Seyboldt)

- Restructure packages (Adrian Seyboldt)

- Some clippy fixes (Adrian Seyboldt)

- Minor style changes (Adrian Seyboldt)


### Ci

- Specify features in CI (Adrian Seyboldt)


## [0.16.0] - 2025-05-27

### Bug Fixes

- Eigen decomposition error for low rank mass matrix (Adrian Seyboldt)


### Miscellaneous Tasks

- Bump arrow version (Adrian Seyboldt)


### Performance

- Replace multiversion with pulp for simd (Adrian Seyboldt)


### Build

- Remove simd_support feature (Adrian Seyboldt)


### Ci

- Add codecov token (Adrian Seyboldt)


## [0.15.1] - 2025-03-18

### Features

- Change defaults for transform adapt (Adrian Seyboldt)


### Miscellaneous Tasks

- Update dependencies (Adrian Seyboldt)

- Update dependencies (Adrian Seyboldt)


### Ci

- Update coverage and audit ci (Adrian Seyboldt)


## [0.14.0] - 2024-12-12

### Documentation

- Update readme (Adrian Seyboldt)


### Features

- Add sampler stats for points (Adrian Seyboldt)

- Adapt some default parameters for transformed adaptation (Adrian Seyboldt)


### Miscellaneous Tasks

- Chore (release): Prepare release (Adrian Seyboldt)

- Update pulp (Adrian Seyboldt)

- Update multiversion (Adrian Seyboldt)


### Refactor

- Remove unnecessary stats structs and add some transform stats (Adrian Seyboldt)


### Testing

- Add vector_dot test (Adrian Seyboldt)

- Fix parallel test in simd mode (Adrian Seyboldt)


## [0.13.0] - 2024-10-23

### Bug Fixes

- Expose adaptation settings (Adrian Seyboldt)

- Append missing values for non-diverging draws (Adrian Seyboldt)

- Fix bug where step size stats were not updated after tuning (Adrian Seyboldt)


### Features

- Bump arrow version (Adrian Seyboldt)

- First part of low rank mass matrix adaptation (Adrian Seyboldt)

- Add option to specify mass matrix update frequency (Adrian Seyboldt)

- Add low-rank modified mass matrix adaptation (Adrian Seyboldt)

- Make cpu_math parallelization configurable (Adrian Seyboldt)

- Add transforming adaptation (Adrian Seyboldt)

- Improve error info for BadInitGrad (Adrian Seyboldt)

- Do not report invalid gradients for transform adapt (Adrian Seyboldt)


### Miscellaneous Tasks

- Update dependencies (Adrian Seyboldt)

- Prepare release (Adrian Seyboldt)

- Update changelog (Adrian Seyboldt)

- Prepare release (Adrian Seyboldt)

- Prepare release (Adrian Seyboldt)

- Update dependencies (Adrian Seyboldt)

- Prepare release (Adrian Seyboldt)


### Refactor

- Switch to arrow-rs (Adrian Seyboldt)

- Refactor mass matrix adaptation traits (Adrian Seyboldt)


### Styling

- Some minor clippy fixes (Adrian Seyboldt)


### Testing

- Add proptest failure and increase tolerance (Adrian Seyboldt)


## [0.9.0] - 2024-04-16

### Documentation

- Update README (Adrian Seyboldt)

- Replace aesara with pytensor in docs (Christian Luhmann)

- Update readme (Adrian Seyboldt)


### Features

- Allow more SIMD in array operations (Adrian Seyboldt)

- Add option to disable NUTS turning check (Adrian Seyboldt)

- Better initial step size estimate (Adrian Seyboldt)

- Add option for draw-based mass matrix estimate (Adrian Seyboldt)

- Parameterize progress callback rate (Adrian Seyboldt)

- Provide sampling duration in sampler progress (Adrian Seyboldt)

- Add index of divergent draws to progress (Adrian Seyboldt)


### Miscellaneous Tasks

- Update changelog (Adrian Seyboldt)

- Update changelog (Adrian Seyboldt)


### Refactor

- Prepare for GPU support (Adrian Seyboldt)

- Move sampler code from nutpie to nuts-rs (Adrian Seyboldt)


### Styling

- Fix formatting (Adrian Seyboldt)


## [0.8.0] - 2023-09-20

### Bug Fixes

- Fix energy error statistic (Adrian Seyboldt)

- Register draws when max treedepth is reached (Adrian Seyboldt)

- Remove incorrect information from README (Adrian Seyboldt)

- Fix compilation without arrow feature (Adrian Seyboldt)

- Correct usage of store_unconstrained (Maxim Kochurov)

- Link in docs to nutpie (Adrian Seyboldt)


### Features

- Add energy_error to sampler stats (Adrian Seyboldt)

- Use symmetric acceptance rate in final window (Adrian Seyboldt)


### Miscellaneous Tasks

- Update Changelog (Adrian Seyboldt)


### Performance

- Increase max and min mass matrix values (Adrian Seyboldt)

- Change default for store_divergences to false (Adrian Seyboldt)


## [0.6.0] - 2023-07-21

### Bug Fixes

- Handle initial zero gradients better (Adrian Seyboldt)


### Features

- Add more sample stats about divergences (Adrian Seyboldt)


### Miscellaneous Tasks

- Update dependencies (Adrian Seyboldt)

- Bump version (Adrian Seyboldt)


## [0.5.1] - 2023-07-03

### Documentation

- Update links from aseyboldt to pymc-devs (Adrian Seyboldt)


### Miscellaneous Tasks

- Update metadata in Cargo.toml (Adrian Seyboldt)

- Add changelog using cliff (Adrian Seyboldt)

- Bump patch version (Adrian Seyboldt)


### Styling

- Formatting fix (Adrian Seyboldt)


## [0.5.0] - 2023-06-29

## [0.4.1] - 2023-01-02

## [0.2.1] - 2022-07-20

<!-- generated by git-cliff -->
