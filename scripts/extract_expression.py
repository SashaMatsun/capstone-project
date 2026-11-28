"""Extract raw counts and CPM by cell and by donor from a converted Kang H5AD."""

import argparse
import logging
from pathlib import Path
import tempfile

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
import yaml


def extract_expression(adata, config):
    """Return cell-level values and sums grouped by donor, condition and cell type."""
    gene = config["gene"]
    cell_types = config["cell_types"]
    if not isinstance(gene, str) or not gene.strip():
        raise ValueError("gene must be a nonempty string.")
    if (not isinstance(cell_types, list) or not cell_types
            or not all(isinstance(value, str) and value.strip() for value in cell_types)
            or len(set(cell_types)) != len(cell_types)):
        raise ValueError("cell_types must be a nonempty list of distinct cell-type labels.")
    conditions = [config["control_condition"], config["stimulated_condition"]]
    if (not all(isinstance(value, str) and value.strip() for value in conditions)
            or conditions[0] == conditions[1]):
        raise ValueError("Control and stimulated conditions must be distinct nonempty labels.")
    columns = [config["donor_column"], config["condition_column"], config["cell_type_column"]]
    if len(set(columns)) != 3:
        raise ValueError("Donor, condition and cell-type columns must be distinct.")
    missing = set(columns) - set(adata.obs.columns)
    if missing:
        raise ValueError(f"Missing annotations: {sorted(missing)}")
    if not adata.var_names.is_unique or not adata.obs_names.is_unique:
        raise ValueError("Gene and cell identifiers must be unique.")
    if gene not in adata.var_names:
        raise ValueError(f"Gene {gene!r} is absent from the H5AD gene identifiers.")
    if "counts" not in adata.layers:
        raise ValueError("The H5AD must contain layers['counts'].")
    donor_column, condition_column, cell_type_column = columns
    for column, requested in ((condition_column, conditions), (cell_type_column, cell_types)):
        missing = set(requested) - set(adata.obs[column].dropna())
        if missing:
            raise ValueError(f"Unknown labels in {column}: {sorted(missing)}")

    selected = (adata.obs[cell_type_column].isin(cell_types)
                & adata.obs[condition_column].isin(conditions))
    subset = adata[selected]
    annotations = subset.obs[columns]
    if annotations.isna().any().any():
        raise ValueError("Selected cells have missing grouping annotations.")
    if annotations.astype(str).apply(lambda col: col.str.strip().eq("")).any().any():
        raise ValueError("Selected cells have empty grouping annotations.")
    observed = set(zip(annotations[cell_type_column], annotations[condition_column]))
    expected = {(cell_type, condition) for cell_type in cell_types for condition in conditions}
    if expected - observed:
        raise ValueError(f"No cells for selected cell-type/condition combinations: {sorted(expected - observed)}")

    counts = subset.layers["counts"]
    values = counts.data if sparse.issparse(counts) else np.asarray(counts)
    if (not np.isfinite(values).all() or (values < 0).any()
            or not np.equal(values, np.floor(values)).all()):
        raise ValueError("counts must contain finite, nonnegative integers.")
    # Sum all input genes before extracting the configured gene.
    totals = np.asarray(counts.sum(axis=1, dtype=np.float64)).ravel()
    if (totals <= 0).any() or not np.isfinite(totals).all():
        raise ValueError("Selected cells must have finite, positive total counts.")
    gene_values = counts[:, [adata.var_names.get_loc(gene)]]
    if sparse.issparse(gene_values):
        gene_values = gene_values.toarray()
    cells = annotations.astype(str).copy()
    cells.columns = ["patient", "condition", "cell_type"]
    cells.insert(0, "cell_id", subset.obs_names.astype(str))
    cells["gene"] = gene
    cells["gene_counts"] = np.asarray(gene_values).ravel().astype(np.float64)
    cells["total_counts"] = totals
    cells["cpm"] = 1_000_000 * cells["gene_counts"] / cells["total_counts"]
    cells = cells.sort_values(["cell_type", "patient", "condition", "cell_id"]).reset_index(drop=True)

    donors = cells.groupby(["patient", "condition", "cell_type", "gene"],
                          sort=True, observed=True, as_index=False).agg(
        n_cells=("cell_id", "size"),
        gene_counts=("gene_counts", "sum"),
        total_counts=("total_counts", "sum"),
    )
    donors["cpm"] = 1_000_000 * donors["gene_counts"] / donors["total_counts"]
    donors["complete_pair"] = (
        donors.groupby(["patient", "cell_type"], observed=True)["condition"]
        .transform("nunique").eq(2)
    )
    if not donors["complete_pair"].all():
        logging.warning("Some donor/cell-type groups lack a condition; complete_pair marks them.")
    return cells, donors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cells-output", type=Path, required=True)
    parser.add_argument("--donors-output", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    paths = [args.input, args.config, args.cells_output, args.donors_output]
    if len({path.resolve() for path in paths}) != len(paths):
        raise ValueError("Input, config, and output paths must all be distinct.")
    with args.config.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("The YAML config must be a mapping.")
    adata = ad.read_h5ad(args.input)
    cells, donors = extract_expression(adata, config)
    # Finish calculations and stage both files before replacing existing outputs.
    with tempfile.TemporaryDirectory() as directory:
        for table, filename in ((cells, "cells.tsv"), (donors, "donors.tsv")):
            table.to_csv(Path(directory) / filename, sep="\t", index=False,
                         float_format="%.12g")
        for filename, destination in (("cells.tsv", args.cells_output),
                                      ("donors.tsv", args.donors_output)):
            destination.parent.mkdir(parents=True, exist_ok=True)
            # A temporary sibling permits atomic replacement on the same filesystem.
            with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
                temporary = Path(stream.name)
            try:
                temporary.write_bytes((Path(directory) / filename).read_bytes())
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
    logging.info("Extracted %s cells and %s donor/condition/cell-type groups for %s.",
                 len(cells), len(donors), config["gene"])
    logging.info("Wrote %s and %s", args.cells_output, args.donors_output)


if __name__ == "__main__":
    main()
