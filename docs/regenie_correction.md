# REGENIE Binary Correction Semantics

`g` mirrors REGENIE-style binary fallback flags instead of exposing a `g`-specific correction enum.

The public CLI shape is:

```bash
g regenie2 --trait-type binary --firth --approx --pThresh 0.01 --firth-se
```

The public Python API shape is:

```python
api.Regenie2BinaryConfig(firth=True, approx=True, p_threshold=0.01, firth_se=True)
```

## Supported Modes

Default binary mode is score-test-only. No fallback correction candidates are generated unless a fallback mode is explicitly requested.

`--firth --approx` is the supported fallback mode. It uses approximate Firth for score-test rows with `LOG10P > -log10(pThresh)`.

`--approx` without `--firth` is accepted with a warning and ignored, matching REGENIE flag semantics.

`--firth --spa` is accepted with a warning and Firth is preferred, matching REGENIE flag precedence.

`--firth` without `--approx` raises `NotImplementedError` until exact Firth is implemented or parity-proven.

`--spa` raises `NotImplementedError` until real SPA fallback is implemented.

`pThresh` must be in `(0, 1)` and defaults to `0.05`.

## Output Semantics

Internal extra codes remain useful for diagnostics:

- `0`: score-test row
- `1`: successful Firth fallback row
- `2`: reserved SPA fallback row
- `3`: failed fallback row

User-facing `EXTRA` output renders codes `0`, `1`, and `2` as null. Code `3` renders as `TEST_FAIL`.

Successful Firth rows therefore do not write `FIRTH` to `EXTRA`; only failed fallback rows write `TEST_FAIL`.
