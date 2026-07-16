# Vendored OpenXLA FFI headers

These three header-only external FFI files were copied byte-for-byte from the
official `jaxlib==0.10.2` Linux wheel installed on 2026-07-16:

- `jaxlib/include/xla/ffi/api/api.h`
  (`sha256:7f76572a80ed2097e5924e6d02d84891c725300172280bd009c0a7c9ac7961eb`)
- `jaxlib/include/xla/ffi/api/c_api.h`
  (`sha256:85fc385c2d3a6b539a05b9cf4c3535aa24b4b41040f9e111c1f2c11b0e2fa539`)
- `jaxlib/include/xla/ffi/api/ffi.h`
  (`sha256:4e4a1d8f9825e88e15a2bcbb7c08eb6233f020b952cab5bbbb8510e3017515c5`)

The headers declare XLA FFI C API version 0.3 and state that this external
interface is header-only with no dependencies beyond the C++ standard library.
Their upstream source paths are under
<https://github.com/openxla/xla/tree/main/xla/ffi/api>. The accompanying
standard Apache License 2.0 text is copied from
<https://github.com/openxla/xla/blob/main/LICENSE>
(`sha256:43070e2d4e532684de521b885f385d0841030efa2b1a20bafb76133a5e1379c1`).
Upstream does not publish a repository-level `NOTICE` file. The jaxlib wheel's
aggregate dependency license is intentionally not vendored because this crate
does not copy those dependencies.

No NVIDIA header or binary is vendored in this crate. Runtime architecture and
deployment behavior are documented with the integration that consumes this
private native capability; this file records only third-party provenance.
