Tiny end-to-end CLI smoke fixture.

The genotype input is `tests/data/bgen/haplotypes.bgen`, which is a four-sample,
four-variant BGEN with embedded sample identifiers `sample_0` through `sample_3`.
The text files in this directory provide phenotype, covariate, and LOCO
prediction inputs for a real `g regenie` run.

The test writes `pred.list` into a temporary directory so the LOCO path can be
absolute. This keeps the committed fixture relocatable and avoids Git LFS.
