# Third-Party Notices

This file contains notices for third-party open-source software included in or
used by this NVIDIA fork of `cell-eval`.

This repository does not vendor third-party dependency source code. Runtime,
optional, build, and development dependencies are installed from package indexes.

## Upstream Project

### cell-eval

- Source: <https://github.com/ArcInstitute/cell-eval>
- License: MIT
- License text: [LICENSE](LICENSE)
- Copyright notice:
  - Copyright (c) 2025 Arc Research Institute
- Notes:
  - This repository is an NVIDIA fork of the upstream Arc Institute project.
  - NVIDIA modifications are subject to the same project license unless a file
    states otherwise.

## Runtime Dependencies

The following packages are declared as runtime dependencies in
`pyproject.toml`. They are installed from package indexes and are not vendored
in this repository.

| Component | Version / Constraint | License | Copyright / Attribution | Source | License Text |
| --- | --- | --- | --- | --- | --- |
| anndata | `>=0.12.10` | BSD-3-Clause | Copyright (c) 2025, scverse; Copyright (c) 2017-2018, P. Angerer, F. Alexander Wolf, Theis Lab | <https://github.com/scverse/anndata> | <https://github.com/scverse/anndata/blob/main/LICENSE> |
| pdex | `>=0.2.2` | MIT | Copyright (c) 2025 Arc Research Institute | <https://github.com/ArcInstitute/pdex> | <https://github.com/ArcInstitute/pdex/blob/main/LICENSE> |
| polars | `>=1.30.0` | MIT | Copyright (c) 2025 Ritchie Vink; Copyright (c) 2024 (Some portions) NVIDIA CORPORATION & AFFILIATES. All rights reserved. | <https://github.com/pola-rs/polars> | <https://github.com/pola-rs/polars/blob/main/LICENSE> |
| pyarrow | `>=18.0.0` | Apache-2.0 | Copyright 2016-2026 The Apache Software Foundation | <https://github.com/apache/arrow> | <https://github.com/apache/arrow/blob/main/LICENSE.txt> |
| PyYAML | `>=6.0.2` | MIT | Copyright (c) 2017-2021 Ingy döt Net; Copyright (c) 2006-2016 Kirill Simonov | <https://github.com/yaml/pyyaml> | <https://github.com/yaml/pyyaml/blob/main/LICENSE> |
| scanpy | `>=1.10.3` | BSD-3-Clause | Copyright (c) 2025 scverse; Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab | <https://github.com/scverse/scanpy> | <https://github.com/scverse/scanpy/blob/main/LICENSE> |
| tqdm | `>=4.67.1` | MPL-2.0 AND MIT | Copyright retained by tqdm authors; notable notices include Casper da Costa-Luis, Google Inc., and Noam Yorav-Raphael | <https://github.com/tqdm/tqdm> | <https://github.com/tqdm/tqdm/blob/master/LICENCE> |

## Optional Dependencies

The following package is declared as an optional dependency and is not installed
by the default `cell-eval` dependency set.

| Component | Version / Constraint | License | Copyright / Attribution | Source | License Text |
| --- | --- | --- | --- | --- | --- |
| igraph | `>=0.11.8` | GNU General Public License; upstream license text is GPL version 2 | Python interface by the python-igraph authors; PyPI metadata identifies Tamas Nepusz as author | <https://github.com/igraph/python-igraph> | <https://github.com/igraph/python-igraph/blob/main/LICENSE> |

## Build and Development Dependencies

The following packages are used for building, testing, or development. They are
not required for normal runtime use unless a downstream distribution explicitly
bundles development tooling.

| Component | Version / Constraint | License | Copyright / Attribution | Source | License Text |
| --- | --- | --- | --- | --- | --- |
| hatchling | build backend | MIT | Copyright (c) 2017-present Ofek Lev | <https://github.com/pypa/hatch/tree/master/backend> | <https://github.com/pypa/hatch/blob/master/LICENSE.txt> |
| ipykernel | `>=6.29.5` | BSD-3-Clause | Copyright (c) 2015, IPython Development Team | <https://github.com/ipython/ipykernel> | <https://github.com/ipython/ipykernel/blob/main/LICENSE> |
| prek | `>=0.4.3` | MIT | Copyright (c) 2024 j178 | <https://github.com/j178/prek> | <https://github.com/j178/prek/blob/master/LICENSE> |
| pytest | `>=8.3.5` | MIT | Copyright (c) 2004 Holger Krekel and others | <https://github.com/pytest-dev/pytest> | <https://github.com/pytest-dev/pytest/blob/main/LICENSE> |
| ty | `>=0.0.19` | MIT | Copyright (c) 2025 Astral Software Inc. | <https://github.com/astral-sh/ty> | <https://github.com/astral-sh/ty/blob/main/LICENSE> |

## External Data and Materials

The source repository and package do not bundle external datasets, model
weights, or third-party binary artifacts. Documentation and tutorial notebooks
may reference external resources, including Virtual Cell Challenge resources,
that are retrieved separately by the user.
