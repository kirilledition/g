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
      in {
        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            uv
            just
            python313
            maturin
            ruff
            zstd
            cargo
            clippy
            rustc
            rustfmt
            pkg-config
            cacert
            openssl
            plink2Package
            regeniePackage
          ];

          shellHook = ''
            export UV_PYTHON=python3.13
            export SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
            export NIX_SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt
            export LD_LIBRARY_PATH=/run/opengl-driver/lib:''${NIX_LD_LIBRARY_PATH:+:$NIX_LD_LIBRARY_PATH}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
            echo "GWAS Engine dev shell ready (uv, Rust, plink2, regenie)."
          '';
        };
      });
}
