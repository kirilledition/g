from __future__ import annotations

import pytest

import scripts.install_nsight_tools as nsight_installer


def test_nsight_installer_selects_newest_systems_package_and_compatible_compute_package() -> None:
    package_index = """
Package: nsight-systems-2025.6.3
Version: 2025.6.3.343-1
Filename: ./nsight-systems-2025.6.3_2025.6.3.343-1_amd64.deb
SHA256: 1111111111111111111111111111111111111111111111111111111111111111

Package: nsight-systems-2026.1.3
Version: 2026.1.3.243-1
Filename: ./nsight-systems-2026.1.3_2026.1.3.243-1_amd64.deb
SHA256: 2222222222222222222222222222222222222222222222222222222222222222

Package: cuda-nsight-compute-12-2
Version: 12.2.2-1
Filename: ./cuda-nsight-compute-12-2_12.2.2-1_amd64.deb
SHA256: 5555555555555555555555555555555555555555555555555555555555555555
Depends: nsight-compute-2023.2.2 (>= 2023.2.2.3)

Package: nsight-compute-2023.2.2
Version: 2023.2.2.3-1
Filename: ./nsight-compute-2023.2.2_2023.2.2.3-1_amd64.deb
SHA256: 6666666666666666666666666666666666666666666666666666666666666666

Package: nsight-compute-2026.1.1
Version: 2026.1.1.2-1
Filename: ./nsight-compute-2026.1.1_2026.1.1.2-1_amd64.deb
SHA256: 3333333333333333333333333333333333333333333333333333333333333333

Package: nsight-compute-2026.2.0
Version: 2026.2.0.7-1
Filename: ./nsight-compute-2026.2.0_2026.2.0.7-1_amd64.deb
SHA256: 4444444444444444444444444444444444444444444444444444444444444444
"""
    packages = nsight_installer.parse_package_index(package_index)

    nsight_systems = nsight_installer.select_package(
        packages=packages,
        tool=nsight_installer.NSIGHT_TOOLS[0],
        requested_package_name=None,
    )
    nsight_compute = nsight_installer.select_package(
        packages=packages,
        tool=nsight_installer.NSIGHT_TOOLS[1],
        requested_package_name=None,
        nsight_compute_cuda_version=nsight_installer.CudaToolkitVersion(major=12, minor=2),
    )

    assert nsight_systems.package_name == "nsight-systems-2026.1.3"
    assert nsight_compute.package_name == "nsight-compute-2023.2.2"


def test_nsight_installer_honors_requested_package() -> None:
    package_index = """
Package: nsight-compute-2026.1.1
Version: 2026.1.1.2-1
Filename: ./nsight-compute-2026.1.1_2026.1.1.2-1_amd64.deb
SHA256: 3333333333333333333333333333333333333333333333333333333333333333

Package: nsight-compute-2026.2.0
Version: 2026.2.0.7-1
Filename: ./nsight-compute-2026.2.0_2026.2.0.7-1_amd64.deb
SHA256: 4444444444444444444444444444444444444444444444444444444444444444
"""
    packages = nsight_installer.parse_package_index(package_index)

    selected_package = nsight_installer.select_package(
        packages=packages,
        tool=nsight_installer.NSIGHT_TOOLS[1],
        requested_package_name="nsight-compute-2026.1.1",
    )

    assert selected_package.package_name == "nsight-compute-2026.1.1"


def test_nsight_installer_fails_when_compatible_compute_package_is_missing() -> None:
    package_index = """
Package: cuda-nsight-compute-13-3
Version: 13.3.0-1
Filename: ./cuda-nsight-compute-13-3_13.3.0-1_amd64.deb
SHA256: 5555555555555555555555555555555555555555555555555555555555555555
Depends: nsight-compute-2026.2.0 (>= 2026.2.0.7)

Package: nsight-compute-2026.2.0
Version: 2026.2.0.7-1
Filename: ./nsight-compute-2026.2.0_2026.2.0.7-1_amd64.deb
SHA256: 4444444444444444444444444444444444444444444444444444444444444444
"""
    packages = nsight_installer.parse_package_index(package_index)

    with pytest.raises(RuntimeError, match="No Nsight Compute package compatible"):
        nsight_installer.select_package(
            packages=packages,
            tool=nsight_installer.NSIGHT_TOOLS[1],
            requested_package_name=None,
            nsight_compute_cuda_version=nsight_installer.CudaToolkitVersion(major=12, minor=2),
        )
