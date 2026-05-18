### 2. Harden sample identity alignment without silently changing behavior

**Current issue**

The new architecture moved sample alignment into Rust, which is good. But the current alignment contract still preserves the older semantics: phenotype and covariate rows are matched by `IID` only, while prediction alignment is more naturally tied to `FID/IID`.

That may be intentional for compatibility, but it is risky when `IID` is not globally unique.

**Do not silently switch the default to `(FID, IID)` yet.** That could break existing workflows.

**Recommended direction**

Keep the current default for compatibility, but make the identity contract explicit.

Add something like:

```python
sample_key_mode: Literal["iid", "fid_iid"] = "iid"
```

or, if you want less public API surface:

```python
strict_sample_identity: bool = True
```

**Implementation guidance**

Phase this in:

### Phase A: validation only

In the current IID-only mode, validate:

* duplicate `IID` in BGEN sample metadata;
* duplicate `IID` in phenotype table;
* duplicate `IID` in covariate table;
* duplicate or missing IDs in prediction sample keys;
* mismatch between aligned phenotype/covariate keys and prediction keys.

For the first release, duplicate-IID handling can be:

```text
error by default
allow only with explicit compatibility flag
```

Example config:

```python
allow_duplicate_iid_alignment: bool = False
```

### Phase B: add explicit `(FID, IID)` mode

Once validation is stable, add a proper `fid_iid` mode. In that mode:

* phenotype table must contain `FID` and `IID`;
* covariate table must contain `FID` and `IID`;
* sample file alignment uses both;
* LOCO prediction alignment uses the same key;
* all aligned arrays are sorted by BGEN sample index after joining.

### Phase C: document the contract

The README should say something like:

```text
By default, sample alignment uses IID and requires IID uniqueness.
For datasets where IID is not globally unique, use sample_key_mode="fid_iid".
```

This turns a hidden data-integrity risk into an explicit user-facing rule.
