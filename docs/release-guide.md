# Release and archive guide

The public release accompanies *Probabilistic modelling of buildings as networks*
in Structural Safety, Article 102783,
[doi:10.1016/j.strusafe.2026.102783](https://doi.org/10.1016/j.strusafe.2026.102783).
Version 1.0.0 covers the stationary model. The article's volume and issue are
not yet recorded in Crossref; do not guess them or a software release date.

## Prepare the candidate

1. Work on `main`. Review the tracked files and history reachable from this
   branch for public suitability.
   The maintainer has selected standard MIT terms, with a separate scholarly
   citation request. ETH Zurich is identified as copyright holder; software
   creators are credited separately in [AUTHORS.md](../AUTHORS.md). The maintainer
   has confirmed approval for the MIT release. Retain the release-approval record
   with the project records.
2. Check the [release notes](release-notes-v1.0.0.md), [citation](../CITATION.cff),
   [contributors](../AUTHORS.md) and [changelog](../CHANGELOG.md). Keep software
   and paper identifiers distinct. `CITATION.cff` is the source of archive metadata.
   Ensure that the README, `LICENSE`, package and citation metadata all identify
   MIT, and that citation guidance does not introduce additional license terms.
3. Confirm the validation workflow passes on the exact candidate commit. Run
   `python scripts/validate_reproduction.py` to check the curated evidence without
   new simulation. Review its computational limits in the reproduction guide.
4. Build the source distribution and wheel with `python -m build`. Inspect the
   archive contents and test the installed wheel outside the checkout. Keep raw
   databases, local outputs and private working notes excluded from public artifacts.

## Publish version 1.0.0

Repository visibility, tagging, release publication and archive publication
require the maintainer's release decision. The steps below describe that process;
they do not indicate that publication has already occurred.

1. Choose the final `main` commit after CI passes. Set the actual release
   date in the changelog and software citation if it is known. If this changes
   the candidate, validate the resulting commit before tagging.
2. Make the repository public when ready and create the `v1.0.0` tag at that
   verified commit. Create a GitHub release using these release notes and attach
   the built source distribution and wheel. Record artifact SHA-256 hashes.
3. Archive that same tagged source in Zenodo using one of the routes below.
   Record the archive DOI, version, source commit and artifact hash together.
4. Add the version-specific software DOI to the citation and README after the
   archive is published. The paper DOI is already recorded separately under
   `preferred-citation`; add volume/issue details when assigned. Keep later
   metadata updates on `main`; do not move an already published tag.

[GitHub releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases)
are based on Git tags, so the tag must identify the verified source revision.

## Archive route

For GitHub integration, enable the repository in Zenodo before publishing the
GitHub release. The existing
[CITATION.cff metadata](https://help.zenodo.org/docs/github/describe-software/citation-file/)
describes the software for the archive. A separate `.zenodo.json` is unnecessary
for this release: Zenodo recommends CFF alone when no Zenodo-specific features
are needed, and `.zenodo.json` would override CFF metadata if both were present.
See [Zenodo metadata guidance](https://help.zenodo.org/docs/github/describe-software/zenodo-json/).

For manual upload, generate a single source ZIP from the verified release tag,
select the Software resource type, review the authors/version/license and publish
the record. Include the tracked reproduction package in that ZIP. The wheel can
remain a GitHub release asset. Zenodo's
[manual software-archive guide](https://help.zenodo.org/docs/github/archive-software/manual-upload/)
uses one compressed source file for compatibility with Software Heritage archiving.

Do not put the article DOI in the software's top-level `doi` field. It belongs
under `preferred-citation` when available. Do not replace version 1.0.0 artifacts
silently; publish a new version for subsequent software changes.
