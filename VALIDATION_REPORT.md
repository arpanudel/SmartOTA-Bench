# SmartOTA-Bench camera-ready artifact validation report

Validation date: 2026-09-03

## Frozen evidence and provenance

The frozen archive
`smartota-final-deterministic-benchmark-20260717-clean-rerun.tar.gz` was
preserved unchanged. Its verified SHA-256 is:

```text
0fb643d7361a537ab7740b1923168ee46548b8ee3ff4be2a24d509e221a644ef
```

The archive contains the historical reported results. Its scientific data was
not altered, regenerated, recompressed, or replaced to address documentation
gaps.

The result-generation source commit is
`2942cbde2fd344cccea23aabbe8d5bf168610c71`. The distinct paper-evidence
promotion commit is `f2de5a2dccf15df89b96e4ca7ab0efc0775f43dc`.
These are separate provenance anchors: the former identifies the source that
generated result rows; the latter identifies their later promotion into the
paper evidence tree.

## Historical documentation boundary

The historical package does not contain `artifact_environment.txt`.
Consequently, the exact historical generation environment cannot be
reconstructed from the archive and has not been fabricated.

Result rows record Python 3.12.7 and selected dependency versions, but that
metadata is not a complete environment lock. Any environment recorded during
the present validation is the **validation environment**, not necessarily the
original generation environment.

Generated visual material in the historical package is under `paper/plots/`,
not `paper/figures/`. The historical `commands_used.md` records the two
benchmark invocations, report generation, and result audit, but it does not
contain complete environment-setup and test commands.

The root `REPRODUCING.md` therefore supplies independently reconstructed and
currently revalidated setup, test, audit, data-preparation, and reproduction
commands. Those additions are release documentation; they are not represented
as a historical execution log or asserted to be the exact original commands.

## Release-time validation scope

The release-time checks cover:

- the focused deterministic benchmark unit suite;
- the frozen result provenance audit against the generation commit;
- archive safety, extraction, internal checksums, and byte equality with the
  convenience extraction;
- the 1,152-row result and attempt ledgers;
- duplicate configuration and run-identifier checks;
- required result-file presence;
- regenerated tables/plots in a temporary output directory;
- release-wide SHA-256 checksums;
- Git whitespace checks;
- credential/private-information scanning;
- large-file inventory; and
- `CITATION.cff` syntax and CFF 1.2.0 schema validation.

Detailed release-time outcomes are recorded after all checks complete in the
local release-preparation commit:

| Check | Outcome |
| --- | --- |
| Archive SHA-256 before copy | `0fb643d7361a537ab7740b1923168ee46548b8ee3ff4be2a24d509e221a644ef` |
| Archive SHA-256 after copy | `0fb643d7361a537ab7740b1923168ee46548b8ee3ff4be2a24d509e221a644ef` |
| `python -m unittest discover -s tests` | PASS — 124 tests, 0 skipped |
| Frozen result audit with expected generation commit | PASS — 1,152 rows, 0 dirty, 0 errors |
| Archive extraction and internal SHA-256 manifest | PASS |
| Extracted archive versus convenience extraction | PASS — byte-for-byte identical |
| Ledger and configuration checks | PASS — 1,152 rows, 0 duplicate keys, 0 duplicate run IDs |
| Required-result-file check | PASS |
| Table/plot regeneration | PASS — 31 temporary outputs; CSV/LaTeX/PNG match; SVG metadata differs as expected |
| `git diff --check` | PASS |
| Credential/private-information scan | PASS for credentials; immutable historical local paths and one `.invalid` synthetic placeholder documented above |
| Large-file audit | PASS — largest file 8,678,059 bytes; none at or above 10 MiB |
| `CITATION.cff` | PASS — `cffconvert` 2.0.0, CFF schema 1.2.0 |

No remote, tag, visibility change, push, or publication action was part of the
pre-publication validation. The software release date selected for the
publication action is `2026-09-03`; it is the public software release date,
not a conference presentation or proceedings date, and is recorded as
`date-released` in `CITATION.cff`.

## Citation metadata boundary

The paper title and complete author list were checked against the publicly
listed SEC 2026 paper: Arpan Bhattacharjee and Weisong Shi,
*SmartOTA-Bench: Replay-Correct and Deployment-Aware Benchmarking of OTA Update
Planning*. No DOI, ORCID, page range, or proceedings publication date was
available to verify during preparation, so none was invented. The CFF
`date-released` value records the public software release date selected for
this publication action (`2026-09-03`), not the SEC conference schedule.
