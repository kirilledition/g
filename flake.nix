{
  description = "GWAS Engine (g) development environment";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            uv
            just
            python314
            maturin
            ruff
            rustup
            cargo
            rustc
            pkg-config
            openssl
            plink2
            regenie
            cudaPackages.cuda_nvcc
            cudaPackages.cudart
          ];

          shellHook = ''
            export UV_PYTHON=python3.14
            export CUDA_PATH=${pkgs.cudaPackages.cuda_nvcc}
            export LD_LIBRARY_PATH=${pkgs.cudaPackages.cudart}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
            echo "GWAS Engine dev shell ready (uv, Rust, plink2, regenie, CUDA runtime/toolchain)."
          '';
        };
      });
}
