# MATF-CMA — JPM 2026

Paper folder for:

> Sepp, A., Hansen, N., and M.A. Kastenholz (2026), *Capital Market Assumptions
> Using Multi-Asset Tradable Factors: The MATF-CMA Framework*, The Journal of
> Portfolio Management, under review (JPM-093244).

Companion papers in this repository: `achievable_sharpe_faj_2026` (FAJ) and the
ROSAA framework (Sepp, Ossa, Kastenholz, JPM 52(4), 2026).

## Layout

```
paper/           current manuscript source (the SSRN copy). The compiled
                 PDF is built locally and is not tracked.
  figures/       exhibit output, written here by replication/ and read by the
                 manuscript with no copy step in between
replication/     exhibit and bootstrap code
presentations/   talks on the framework                        [untracked]
private/         journal submissions and editor correspondence [untracked]
  v1/            original submission, 14 Mar 2026  (IIJ-JPM-S-26-00105)
  v2/            first revision, 17 May 2026       (JPM-093244_R1)
  reports/       replies to the editor and the editor's letters
  drafts/        intermediate manuscripts between submissions
  sections/      standalone derivation sources feeding the appendices
```

`private/` and `presentations/` are excluded in the repository `.gitignore`.
The manuscript stays out of the public domain until the journal releases it.

## Conventions

- `paper/matf_cma_paper.tex` is the live source. It sets
  `\graphicspath{{figures/}}` and calls figures by bare filename. The scripts in
  `replication/` write their exhibits into `paper/figures/`, so regenerating an
  exhibit changes the compiled paper directly.
- Figure sets frozen at a submission live in `private/v1/figures` and
  `private/v2/figures` and are never overwritten by a regeneration. Each set is
  pruned to the files its own `\includegraphics` calls name, so an unreferenced
  image in the archive folder is not carried forward.
- Every figure traces to a named script in `replication/`. The mapping is in
  `replication/README.md`.
- Production input data (proprietary workbook and CSV pipeline outputs) is not
  committed. The methodology in `replication/README.md` is self-contained.

## Version history

| folder | date | manuscript source | figure set | journal reference |
|---|---|---|---|---|
| `private/v1` | 14 Mar 2026 | `cma_paper_v1.tex` | `Figures/` | IIJ-JPM-S-26-00105 |
| `private/v2` | 17 May 2026 | `cma_paper_v4.tex` | `ArticleFigures/` | JPM-093244_R1 |
| `paper/` | current | `cma_paper_r2_draft.tex` | `exhibits/` -> `paper/figures/` | R2, in preparation |

The figure-set column names the folder those files came from in the Working
Papers archive, so an older compile can be reconstructed.

## Data freeze

The current source is built on the 2026-Q2 production cut, 18 assets, USD,
private equity admitted at w = 0.5. Bootstrap results use B = 500 draws, seed
42, window Jul 2001 to Jun 2026.
