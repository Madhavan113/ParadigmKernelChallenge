# kerneloptimizer

Entries for the [Open Kernel Challenge](https://openkernelchallenge.dev) (Muse Spark Open on H200).

Each `muse_*` directory is one problem as written by `okc init`: `submission.py` is the kernel,
`baseline.py` SGLang's, `reference.py`/`checks.py` the judge's checks, `TASK.md` the spec.
`tools/` holds local test harnesses (Triton CPU interpreter runs inside the judge's image).
