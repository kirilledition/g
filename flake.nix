{
  description = "GWAS Engine (g) development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    # Provides exact Rust toolchain from rust-toolchain.toml (1.96.0 + clippy + rustfmt).
    # This keeps nix develop in sync with the project's pinned Rust version instead of
    # whatever rustc nixpkgs happens to ship.
    rust-overlay.url = "github:oxalica/rust-overlay";
  };

  outputs = { self, nixpkgs, flake-utils, rust-overlay }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [ (import rust-overlay) ];
        };

        # Exact toolchain matching rust-toolchain.toml (channel 1.96.0 + clippy, rustfmt).
        # Using the overlay ensures `nix develop` can build the maturin extension and pass
        # `just check` / `cargo clippy` etc. without "rustc too old" errors.
        rustToolchain = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        plink1Package = pkgs.stdenvNoCC.mkDerivation {
          pname = "plink";
          version = "1.90b7.11";

          src = pkgs.fetchzip {
            url = "https://s3.amazonaws.com/plink1-assets/plink_linux_x86_64_20250819.zip";
            hash = "sha256-NSjHOUMPFrhiv33UGQ1+z/9AeTYRbs7YgUCLd+Y07qA=";
            stripRoot = false;
          };

          installPhase = ''
            runHook preInstall
            mkdir -p "$out/bin" "$out/share/doc/plink"
            install -m755 plink "$out/bin/plink"
            if [ -f LICENSE ]; then
              install -m644 LICENSE "$out/share/doc/plink/"
            fi
            runHook postInstall
          '';
        };

        plink2Package = pkgs.stdenvNoCC.mkDerivation {
          pname = "plink2";
          version = "2.0.0-a.6.33";

          src = pkgs.fetchzip {
            url = "https://s3.amazonaws.com/plink2-assets/alpha6/plink2_linux_amd_avx2_20260228.zip";
            hash = "sha256-VEu1NJ1mTel3wKRZgnm0dCHylN5vjbw0x9hPRbTpaXQ=";
            stripRoot = false;
          };

          installPhase = ''
            runHook preInstall
            mkdir -p "$out/bin" "$out/share/doc/plink2"
            install -m755 plink2 "$out/bin/plink2"
            install -m755 vcf_subset "$out/bin/vcf_subset"
            if [ -f intel-simplified-software-license.txt ]; then
              install -m644 intel-simplified-software-license.txt "$out/share/doc/plink2/"
            fi
            runHook postInstall
          '';
        };

        regeniePackage = pkgs.stdenvNoCC.mkDerivation {
          pname = "regenie";
          version = "4.1";

          src = pkgs.fetchzip {
            url = "https://github.com/rgcgithub/regenie/releases/download/v4.1/regenie_v4.1.gz_x86_64_Linux.zip";
            hash = "sha256-jsjwokhHqGWWcwMWOAWPluDhJkEkkyo4w4y1OsMZ3UI=";
            stripRoot = false;
          };

          nativeBuildInputs = [ pkgs.patchelf ];
          buildInputs = [ pkgs.stdenv.cc.cc.lib ];
          dontUnpack = true;

          installPhase = ''
            runHook preInstall
            mkdir -p "$out/bin"
            install -m755 "$src/regenie_v4.1.gz_x86_64_Linux" "$out/bin/regenie"
            patchelf \
              --set-interpreter "${pkgs.stdenv.cc.bintools.dynamicLinker}" \
              --set-rpath "${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib ]}" \
              "$out/bin/regenie"
            runHook postInstall
          '';
        };

        # CUDA packages from nixpkgs (CUDA 12.9 on nixos-unstable at time of writing).
        # Used only by the gpu dev shell. The JAX CUDA wheels (jax[cuda12] + nvidia-*)
        # provide their own CUDA runtime libraries; we expose these plus the host NVIDIA
        # driver (via /run/opengl-driver on NixOS) so that JAX can see a modern card.
        cudaPackages = pkgs.cudaPackages;

        # Base packages shared by CPU and GPU dev shells.
        # Rust bits come from rustToolchain (pinned via rust-overlay + rust-toolchain.toml)
        # so that maturin builds, cargo test, clippy, and rustfmt all use exactly 1.96.0.
        basePackages = with pkgs; [
          uv
          just
          zstd
          zlib
          rustToolchain
          cargo-llvm-cov
          pkg-config
          cacert
          openssl
          python314
          openjdk11_headless
          openblas
          lapack
          poppler-utils
          plink1Package
          plink2Package
          regeniePackage
        ];

        # CUDA packages from nixpkgs.
        # We define them for documentation / optional manual use (e.g. if you want
        # a full nvcc + cudnn dev environment outside of the JAX wheels).
        # We deliberately do NOT include the heavy ones (cudnn, nccl, full cuda_nvcc
        # etc.) in the gpu devShell by default.
        #
        # Why: For this project the GPU path is JAX + the official nvidia-* wheels
        # that `uv sync --group gpu` installs (jax[cuda12]). Those wheels ship their
        # own CUDA user-space libraries. The *only* thing that must come from the
        # host on NixOS is the NVIDIA driver (libcuda.so.1), which lives at
        # /run/opengl-driver/lib thanks to your system nvidia configuration.
        #
        # Including the big cudaPackages list in the shell closure forces nix to
        # download + set up many GB of CUDA tarballs on first `nix develop .#gpu`.
        # That is what you were seeing as the long "nvidia driver" builds.
        # It is not necessary for running `g regenie --g-device gpu`.
        gpuPackages = [ ];   # keep empty for fast entry; see comment above

        # Common shell initialization shared by both shells.
        # Note: The project's *runtime* Python (including jax[cuda12] and the venv)
        # is managed by uv (see `uv python install` + `uv sync`). We set a version
        # preference here so that `uv run` / `uv sync` default to 3.14 without extra
        # flags. The concrete interpreter for a given command is resolved by uv.
        commonShellHook = ''
          export UV_PYTHON="''${UV_PYTHON:-3.14}"
          export PYO3_PYTHON="''${PYO3_PYTHON:-python3.14}"
          export CFLAGS="-D_DEFAULT_SOURCE ''${CFLAGS:+$CFLAGS}"
          export CXXFLAGS="-D_DEFAULT_SOURCE ''${CXXFLAGS:+$CXXFLAGS}"
          export JAVA_HOME=${pkgs.openjdk11_headless}
          export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
          export NIX_SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
        '';

        # CPU-only dev shell hook (keeps the original lightweight default).
        defaultShellHook = ''
          ${commonShellHook}
          export LD_LIBRARY_PATH=${pkgs.python314}/lib:/run/opengl-driver/lib:''${NIX_LD_LIBRARY_PATH:+:$NIX_LD_LIBRARY_PATH}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
          echo "GWAS Engine dev shell ready (uv, Rust, plink, plink2, regenie)."
          echo "Python for the project is managed by uv (uv python install + uv sync)."
          echo "For NVIDIA GPU + JAX CUDA on NixOS, use:  nix develop .#gpu"
        '';

        # GPU dev shell hook.
        # The important bit on NixOS is making sure the host NVIDIA *driver*
        # (libcuda.so.1 from your system's hardware.nvidia config) is visible
        # to the uv-installed jax[cuda12] + nvidia-* wheels.
        # We do *not* pull the huge cudaPackages here (see the gpuPackages comment).
        gpuShellHook = ''
          ${commonShellHook}
          # Only the host driver is required for the wheels + JAX to talk to your GPU.
          # (The nvidia-*-cu12 packages that come with `uv sync --group gpu` provide
          # their own CUDA runtime/user libraries.)
          driverLibDir="/run/opengl-driver/lib"
          pythonLibDir="${pkgs.python314}/lib"

          # Make the driver discoverable early. This is the same path the project's
          # scripts/probe_jax_runtime.py and g/jax_setup.py already look for.
          export LD_LIBRARY_PATH="''${pythonLibDir}:''${driverLibDir}''${NIX_LD_LIBRARY_PATH:+:$NIX_LD_LIBRARY_PATH}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

          # CUDA_HOME / CUDA_PATH are usually unnecessary when using the official
          # jax[cuda12] wheels. If you later need nvcc/cudnn from nixpkgs for
          # other work, you can add them temporarily or use a one-off nix shell.

          echo "GWAS Engine GPU dev shell ready (uv, Rust, plink, plink2, regenie)."
          echo "NixOS NVIDIA driver visible at /run/opengl-driver/lib."
          echo "JAX GPU support will come from the uv-installed jax[cuda12] wheels (after bootstrap-gpu)."
          echo ""
          echo "Python for the project (jax[cuda12], venvs, etc.) is managed by uv."
          echo "Next steps inside this shell (matches the cluster bootstrap-gpu flow):"
          echo "  just bootstrap-gpu              # uv python install 3.14 + uv sync --group gpu"
          echo "  just doctor-jax                 # probe JAX platform and devices (should see GPU)"
          echo "  just regenie2-binary-gpu-smoke  # quick chr22 binary step2 on GPU (local data/1kg_chr22_full.*)"
          echo ""
          echo "The committed chr22 fixtures + baselines are already present under data/."
          echo ""
          echo "NOTE: this .#gpu shell is now lightweight (no full nixpkgs cudaPackages)."
          echo "      The long CUDA 12.9 downloads you saw earlier are no longer required."
        '';
      in {
        devShells = {
          # Default: full CPU toolchain, suitable for format/lint/typecheck/test-cpu.
          # Does not pull large CUDA derivations.
          default = pkgs.mkShell {
            packages = basePackages;
            shellHook = defaultShellHook;
          };

          # GPU-capable shell for personal NixOS machines with NVIDIA.
          # Enter with:  nix develop .#gpu
          #
          # This shell is intentionally lightweight. It gives you uv, the pinned
          # Rust 1.96, just, the baseline tools, openblas, etc. + the host NVIDIA
          # driver on LD_LIBRARY_PATH.
          #
          # GPU support for this repo comes from:
          #   just bootstrap-gpu   (or just install-gpu-dependencies)
          # which does `uv sync --group gpu` → jax[cuda12] + the official nvidia wheels.
          #
          # We deliberately avoid depending on the heavy nixpkgs cudaPackages here
          # (cudnn, full cuda toolkit, ...) because they are not needed for the
          # JAX path and cause painful multi-GB first-time downloads/builds.
          # (See the comment next to the gpuPackages definition.)
          gpu = pkgs.mkShell {
            packages = basePackages;   # deliberately not adding gpuPackages
            shellHook = gpuShellHook;
          };
        };
      });
}
