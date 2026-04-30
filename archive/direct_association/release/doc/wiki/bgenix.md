`bgenix` is a tool to
create an index of variants in a bgen file and to use that index for efficient retrieval of data
for specific variants or regions.

# Quick start #

To use bgenix with your bgen file (say `myfile.bgen`) you use the following process.

 - first, use `bgenix -index -g myfile.bgen` to create an index file.  The index file will be named `myfile.bgen.bgi`
 - then, use `bgenix -g myfile.bgen` with additional options below to extract ranges of variants.

Here's a quick list of common bgenix options and what they do:

|: Command line or option				|: What it does |
| ----------------------				| ------------- |
| `-help` 									| Print help on the various options bgenix supports |
| `-g`										| Specify the bgen file to operate on. |
| `-index` 									| Create an index file for the given bgen file.  It will be named `file.bgen.bgi` |
| `-incl-rsids` 							| Only output variants that have one of the given rsid(s). |
| `-excl-rsids` 							| Only output variants that don't have the given rsid(s). |
| `-incl-range` 							| Only output variants in one of the given ranges. |
| `-excl-range` 							| Only output variants outside the given ranges.  |
| `-vcf` 									| Transcode data to VCF format. |
| `-v11` 									| Transcode data to BGEN v1.1 format. |
| `-list`									| Don't output genotype data, just list the variants in the index. |

For example, assuming the index file already exists, the command

```
bgenix -g file.bgen -list -incl-range 11:3500000-6500000
```
will print a list of all variants in the given range, while
```
bgenix -g file.bgen -vcf -incl-range 11:3500000-6500000
```
will output a VCF file for that region.

Note that `bgenix` always writes its output to stdout.  You'll therefore usually want to capture this
by redirecting the output to a file like this:
```
bgenix -g file.bgen -incl-range 11:3500000-6500000 > output.bgen
```
or piping to another command lke this:
```
bgenix -g file.bgen -incl-range 11:3500000-6500000 | qctool -g - -filetype bgen -snp-stats -osnp stats.txt
```

See `bgenix -help` for a full list of supported options.

---

# Detailed usage notes #

## Building an index ##

Use

```
bgenix -g myfile.bgen -index
```

to build an index file.  This is typically pretty quick but might take a few minutes on a very large file.

## The index file format ##

`bgenix` index files store the chromosome, position, alleles and identifer of each variant in the bgen file,
along with an byte offset into the bgen file itself so that the variant can be quickly retrieved.
The index file is a [sqlite3](http://www.sqlite.org) file, which means you can inspect (or alter) it using `sqlite3`.

For example you can get a list of variants:
```sh
sqlite3 myfile.bgen.bgi "SELECT * FROM Variant"
```

(But see another way to do this below.)

It's also easy to load this data into programming languages - for example using `pandas.read_sql` in python, or in R:

```R
library( RSQLite )
connection = dbConnect( RSQLite::SQLite(), "myfile.bgen.bgi" )
index = dbGetQuery( connection, "SELECT * FROM Variant" )
```

The full index file format is described on the wiki page [The bgenix index file format]($ROOT/wiki/The bgenix index file format ).

**Note**: For performance reasons bgenix uses ["WITHOUT ROWID"
tables](https://www.sqlite.org/withoutrowid.html) to implement the index. This means you need
sqlite3 version 3.8.2 or greater to inspect the file - otherwise you'll get a message like *"Error:
malformed database schema"*". As an alternative, you can use the `-with-rowid` option when building
the index, which will then be compatible with earlier versions of `sqlite3`.

## Listing variants ##

One of the first things you might want to do after indexing is get a list of variants in the file (or perhaps in a particular region).
If the `-list` option is given, bgenix will do this, returing a list of variants.
For example, using the file `complex.bgen` included in the `example/` folder in the bgen repository, the command:

`bgenix -g example/complex.bgen -list`

produces this output:
```
 # bgenix: started 2016-07-06 09:01:15
alternate_ids	rsid	chromosome	position	number_of_alleles	first_allele
.	V1	01	1	2	A	G
V2.1	V2	01	2	2	A	G
.	V3	01	3	2	A	G
.	M4	01	4	3	A	G,T
.	M5	01	5	2	A	G
.	M6	01	7	4	A	G,GT,GTT
.	M7	01	7	6	A	G,GT,GTT,GTTT,GTTTT
.	M8	01	8	7	A	G,GT,GTT,GTTT,GTTTT,GTTTTT
.	M9	01	9	8	A	G,GT,GTT,GTTT,GTTTT,GTTTTT,GTTTTTT
.	M10	01	10	2	A	G
 # bgenix: success, total 10 variants.
```

(We describe below another way to list variants - by querying the index directly using `sqlite3`.)

## Outputting genotypes ##

By default genotype data is output in the same format as in the input.  This makes `bgenix` fast as it doesn't have to do any processing of the data.

E.g. in the command

```
bgenix -f myfile.bgen -incl-range 1:0-10
```

`bgenix` simply outputs an appropriate BGEN header, and then copies bytes from the input file to the output.

## Transcoding ##

`bgenix` can also transcode data to two other formats:

- to VCF format, enabled with the option `-vcf`
- to BGEN v1.1 format, enabled with the option `-bgen_v1.1`.  The compression level can be altered with the `-compression-level` option.

Currently BGEN v1.1 output is only supported when the input data is in a specific format, namely BGEN with 'layout=2' blocks, 8-bit probability encoding, and all samples are diploid.


## Querying variants ##

`bgenix` can restrict the output based on chromosome and position, or by variant identifier. In general, a
variant will be output if it satisfies at least one of the *inclusion* (`-incl-*`) options, and does
not satisfy any of the *exclusion* (`-excl-*`) options passed on the command-line.

The relavant options are:

|     Syntax      | Notes | Example(s) |
| --------------- | ----- | ------- |
| `-incl-rsids` | Only output variants that have one of the given rsid(s). | `-incl-rsids rs8176719 myids.txt` |
| `-excl-rsids` | Only output variants that don't have the given rsid(s) | `-excl-rsids rs8176719 myids.txt` |
| `-incl-range` | Only output variants in one of the given ranges. | `-incl-range 11:0-1000`, `-incl-range ranges.txt` |
| `-excl-range` | Only output variants outside the given ranges.  | `-excl-range 11:0-1000`, `-excl-range ranges.txt` |

For convenience the above options below take either values directly on the command line, or filenames. If
the argument is a valid filename the file will be opened and values (IDs or ranges) read from it.
`bgenix` expects these files to contain a whitespace-separated list of IDs or chromosome ranges.

Ranges can either be specified by

- A chromosome and two positions (e.g. `11:0-1000`).  This is a closed interval containing both endpoints.
- A chromosome and a starting position (e.g. `1:1000-` or `11:-1000).  These are one-sided intervals.

---

Acknowledgements
====
`bgenix` is motivated by and in some respects designed to mimic [tabix](http://www.htslib.org/doc/tabix.html), the htslib tool for indexing tab-delimited files.  The key functionality of `bgenix` is all implemented using the [sqlite3](http://www.sqlite.org) library.  Thank you, sqlite authors!

