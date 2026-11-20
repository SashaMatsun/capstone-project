"""Convert the Kang SingleCellExperiment RDS to a validated AnnData H5AD."""

import argparse
import logging
from pathlib import Path
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
from rds2py import parse_rds, read_rds
from scipy import sparse


def read_matrix_names(source):
    """Recover ordered labels from the Kang RDS counts matrix itself.

    rds2py 0.10.6 does not propagate this file's rowRanges metadata into
    sce.row_data. The count matrix retains its original R Dimnames.
    """
    raw = parse_rds(str(source))
    attributes = raw["attributes"]
    assays = attributes["assays"]["attributes"]["data"]["attributes"]["listData"]
    assay_names = assays["attributes"]["names"]["data"]
    matrix = assays["data"][assay_names.index("counts")]["attributes"]
    genes, cells = [pd.Index(axis["data"]) for axis in matrix["Dimnames"]["data"]]
    if (len(genes), len(cells)) != tuple(matrix["Dim"]["data"]):
        raise ValueError("Count matrix labels do not match its dimensions.")
    for names in (genes, cells):
        if not names.is_unique or names.isna().any():
            raise ValueError("Original matrix labels must be unique and complete.")
        if not all(isinstance(name, str) and name.strip() for name in names):
            raise ValueError("Original matrix labels must be nonempty strings.")
    gene_ranges = attributes["rowRanges"]["attributes"]
    range_names = gene_ranges["partitioning"]["attributes"]["NAMES"]["data"]
    if list(genes) != list(range_names):
        raise ValueError("Matrix gene order differs from the gene-range metadata.")
    return genes, cells


def validate(adata):
    """Reject unusable identifiers, missing grouping labels, and invalid counts."""
    if not adata.n_obs or not adata.n_vars:
        raise ValueError("The expression matrix is empty.")
    if not adata.obs_names.is_unique or not adata.var_names.is_unique:
        raise ValueError("Cell and gene identifiers must be unique.")
    for names in (adata.obs_names, adata.var_names):
        if names.isna().any() or (names.astype(str).str.strip() == "").any():
            raise ValueError("Cell and gene identifiers must be nonempty.")
    for column in ("patient", "condition", "cell_type"):
        if column not in adata.obs:
            raise ValueError(f"Missing cell annotation: {column}")
        labels = adata.obs[column]
        if labels.isna().any() or labels.astype(str).str.strip().eq("").any():
            raise ValueError(f"Missing labels in {column}.")
    values = adata.layers["counts"].data
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Counts must be finite and nonnegative.")
    if not np.equal(values, np.floor(values)).all():
        raise ValueError("The counts assay contains fractional values.")


def convert(source, destination):
    if source.resolve() == destination.resolve():
        raise ValueError("Input and output must have different paths.")
    logging.info("Reading %s", source)
    genes, cells = read_matrix_names(source)
    sce = read_rds(str(source))
    adata, _ = sce.to_anndata()
    if "counts" not in adata.layers:
        raise ValueError(f"Expected a named counts assay; found {list(adata.layers)}.")
    counts = sparse.csr_matrix(adata.layers["counts"])
    counts.sum_duplicates()
    counts.sort_indices()
    original = sparse.csr_matrix(sce.assays["counts"].T)
    if counts.shape != original.shape or (counts != original).nnz:
        raise ValueError("RNA counts changed during RDS-to-AnnData conversion.")
    if list(adata.obs_names) != list(cells):
        raise ValueError("Cell identifiers changed during conversion.")
    if len(genes) != adata.n_vars:
        raise ValueError("Number of gene labels differs from the converted matrix.")
    adata.var_names = genes
    adata.var["name"] = genes.to_numpy()
    adata.layers["counts"] = counts
    # X and the named layer both contain raw counts at this conversion stage.
    adata.X = counts.copy()
    validate(adata)
    if adata.shape != tuple(reversed(sce.shape)):
        raise ValueError("Unexpected cell/gene dimensions after conversion.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        temporary = Path(directory) / "converted.h5ad"
        adata.write_h5ad(temporary, compression="gzip")
        saved = ad.read_h5ad(temporary)
        validate(saved)
        if list(saved.var_names) != list(genes) or list(saved.obs_names) != list(cells):
            raise ValueError("Original matrix labels changed during H5AD serialization.")
        pd.testing.assert_frame_equal(adata.obs, saved.obs, check_dtype=False,
                                      check_categorical=False)
        pd.testing.assert_frame_equal(adata.var, saved.var, check_dtype=False,
                                      check_categorical=False)
        if (counts != saved.layers["counts"]).nnz or (counts != saved.X).nnz:
            raise ValueError("RNA counts changed during H5AD serialization.")
        temporary.replace(destination)

    logging.info("Saved %s (%s cells x %s genes)", destination, adata.n_obs, adata.n_vars)
    logging.info("Donors: %s", adata.obs["patient"].nunique())
    logging.info("First gene identifiers: %s", adata.var_names[:10].tolist())
    logging.info("Conditions: %s", sorted(adata.obs["condition"].unique().tolist()))
    logging.info("Cell types:\n%s", adata.obs["cell_type"].value_counts().to_string())
    logging.info("Passed count and annotation round-trip checks.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
