# Put this prepared snapshot on GitHub

Suggested repository name: `biot-neural-operators`.
This archive has not created a GitHub repository or pushed any files.
Start with a private repository while completing provenance/license review and
checking this snapshot against the actual scripts used on the Linux workstation.

1. Extract the archive. Open a terminal inside the `biot-neural-operators` folder.
2. Create an empty repository on your GitHub account, with no generated README,
   license or gitignore (this archive supplies the source and documentation).
3. Run the commands below, replacing YOUR_ACCOUNT with your GitHub account:

```bash
git init -b main
python tools/check_source_snapshot.py
git add .
git status --short
git commit -m "Add Biot DG solvers, dataset workflows and neural operator experiments"
git remote add origin https://github.com/YOUR_ACCOUNT/biot-neural-operators.git
git push -u origin main
```

Use your normal GitHub authentication. Do not paste tokens into source files.
Review the staged filenames before committing; datasets, weights and generated
results are ignored. `SOURCE_MANIFEST.json` captures the packaged source, and
will need deliberate updates when the code changes. After publication, add the
real repository URL to `CITATION.cff`; no placeholder URL is included there.

A future release should record a tested environment and quantitative solver
validation results, then tag a version and optionally archive it with a DOI.

## Publication decisions

Confirm the GitHub account or organization, repository name and public/private
visibility. `biot-neural-operators` is a proposed name, not an existing remote.
A private staging repository is a practical first step while source ownership
and license intent are unresolved. Public visibility and an open-source license
are separate choices. No license is selected in this snapshot; confirm the
maintainer's intended terms before adding any LICENSE file.

Review docs/REVIEW_REPORT.md before uploading. Run all-file syntax checking with
`python -m compileall -q .` in addition to the current snapshot checker.
The current inventory covers the reviewed files except the two inventories
themselves. SOURCE_MANIFEST.json includes available original paths and hashes;
REVIEW_MANIFEST.json inventories all files except itself. Regenerate current
hashes deliberately after edits, preserving historical source_sha256 metadata.

After uploading, inspect the actual Actions run, record its result, and replace
pre-publication status text with the real repository URL and publication state.
Add `repository-code` to CITATION.cff; set `date-released` only for an actual
release. A paper, DOI, trained checkpoint and completed benchmark must never be
implied by the presence of a GitHub repository.
