# Migdal

Migdal is a compact scientific Python package for comparing radial one-band
G0W0, GW0, scGW, and Takada projected pairing calculations for an
SrTiO3-like model.

```{note}
The source code and runnable scripts will be released together with the
accepted article.
```

The documentation is intentionally staged. Start with a runnable
{doc}`example <examples>`, then read the {doc}`theory` page, and use the
workflow, parameter, output, and troubleshooting pages as needed.

```{toctree}
:hidden:
:maxdepth: 2

quickstart
theory
workflows
examples
parameters
output
cli
api
troubleshooting
```

## Start here

| Page | Use it for |
| --- | --- |
| {doc}`quickstart` | Install, check the optimized kernels, and run the smallest tutorial. |
| {doc}`theory` | Understand the physical model, main formulas, numerical reductions, and references. |
| {doc}`workflows` | Choose a method and run the common calculation paths. |
| {doc}`examples` | Choose the next runnable script and understand its outputs. |
| {doc}`parameters` | Look up driver settings, units, and controls. |
| {doc}`output` | Interpret result rows, CSV columns, and convergence flags. |
| {doc}`cli` | Run compact package-level density scans and manage sparse-IR caches. |
| {doc}`api` | Write Python code around the public drivers. |
| {doc}`troubleshooting` | Diagnose installation, convergence, grid, and cache issues. |
