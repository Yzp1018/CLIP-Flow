# Historical sources

These files preserve the supplied archive bytes, including inactive branches, hard-coded server paths and experimental losses. They are evidence and reference implementations, not the supported experiment entry points.

- `original/`: Python sources from `cflow-ad-master.zip` relevant to CLIP-Flow and its imported modules.
- `cflowtest/`: the supplied 19-dataset evaluation entry point and all six result text files, plus its data loader and support sources.
- `UniversalFake/`: the comparison baseline sources supplied inside the first archive. Its original dependencies and environment are separate from the consolidated CLIP-Flow package. It is preserved without claiming that its modified training code reproduces published UnivFD results.

`docs/ARCHIVE_MANIFEST.json` records archive paths and SHA-256 checksums. The shared CLIP tokenizer vocabulary is also preserved where the baseline expects it. Use `python -m clipflow`, rather than executing these historical scripts, for the consolidated implementation.
