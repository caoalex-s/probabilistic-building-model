# Changelog

All notable changes to the public stationary release are documented here.

## [1.0.0] - 2026-10-16

### Release preparation

- Preserve the exact saved-evidence bytes across operating systems and validate
  the evidence hashes and scientific calculations in CI.

- Restore package metadata, CLI entry points, SQL package data, and pinned paper dependencies.
- Align the analysis documentation and reproduction settings with the R1 manuscript.
- Add regressions using retained publication counts and distinguish synthetic estimator tests.
- Remove unused legacy staging modules outside the stationary model contract.
- Make notebook paths portable, default to standard plots, and clear saved execution outputs.
- Align damage-notebook thresholds/adverse states and preserve existing campaign results.
- Correct citation metadata and add manuscript contributor roles.
- Add automated tests and distribution builds across Linux, Windows, and macOS.
- Add portable fresh-campaign commands and simulation-free validation of archived counts.
- Include reviewed Figures 3–5, full scientific count data, provenance and SHA-256 hashes.
- Verify Figure 3 with reduced samples and an independent seed; reproduce damage
  scenarios at the manuscript sample count and check the risk-equivalence root.
- Validate source and wheel installation, packaged SQL, and commands outside the checkout.
- Prepare version 1.0.0 release notes and software archive instructions.
- Restructure the README around installation, a small example, equations and
  repository/notebook navigation; include direct paper and software citations.
- Add an introductory eight-compartment notebook and preserve advanced usage
  in a separate reference guide. Use GitHub-compatible Markdown math delimiters.
- Retain the standard MIT license, permitting commercial reuse, and present
  paper/software citation as a separate scholarly request. Remove the superseded
  custom-license proposal from public documentation. Identify ETH Zurich as
  copyright holder following the maintainer's confirmation of release approval,
  and retain separate software authorship and institutional acknowledgments.
- Remove duplicate top-level SQL schemas and the builder's obsolete external
  writer import. Explicit source-root schema lookup uses `pbm/resources/sql/`;
  the former unpacked `resources/sql/` layout is no longer accepted.
- Remove local-working-material references from public guides and record the
  development history across ETH, Empa and NTU, the concept's origins during
  a Lund research stay, and Asim Kumalic's master's-thesis code contribution.
- Explicitly acknowledge ETH Zurich, Empa and Nanyang Technological University
  in the README and contributor notes, separately from authorship and copyright.
- Add the published article DOI (10.1016/j.strusafe.2026.102783) and article
  number 102783 to the citation and release documentation, and link the paper
  from the package metadata. Update the recorded validation status after cleanup.

## [1.0.0] - Release date pending

### Scope

Version 1.0 implements the model associated with
*Probabilistic modelling of buildings as networks*.
