# Negative `SYMMETRY_FUSED` checks

`CH4_gxtb_kp_symmetry_fused_batch_zero.inp` is expected to terminate with a nonzero status and
`IMAGE_BATCH_SIZE must be positive`. CP2K's regression driver treats every nonzero exit status as a
runtime failure, so this fixture is kept outside `TEST_FILES.toml` and is exercised explicitly by
the qualification manifest command.
