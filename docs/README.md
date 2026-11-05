# Documentation

| Document | Answers |
|---|---|
| [../README.md](../README.md) | What the project is, its main numbers, and how to start |
| [INSTALL.md](INSTALL.md) | How to install from zero: ROCm, the installer, weights, a service, removal |
| [CONFIGURATION.md](CONFIGURATION.md) | Options of `serve.sh`, presets, own presets, choosing settings, using the API |
| [PERFORMANCE.md](PERFORMANCE.md) | All measurements with their configuration: models, all-reduce, llama.cpp, pipeline parallelism, quality, repeats |
| [ARCHITECTURE.md](ARCHITECTURE.md) | What the patch changes in vLLM and why: all-reduce, kernels, attention, MTP |
| [SWITCHES.md](SWITCHES.md) | Every environment variable added by the fork |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | Open problems, workarounds and what was fixed |
| [TROUBLESHOOTING.md](TROUBLESHOOTING.md) | What to do when something fails: start, first request, wrong output, speed, memory, stuck processes |
| [RELATED_WORK.md](RELATED_WORK.md) | Other public projects on gfx1030 |
| [notes/README.md](notes/README.md) | Working notes in Ukrainian: the history of measurements and decisions |

Next to the code:

| Document | Answers |
|---|---|
| [../bench/README.md](../bench/README.md) | How to run the benchmarks |
| [../results/README.md](../results/README.md) | Format of the measurement records, how tables are generated |
| [../kernels/README.md](../kernels/README.md) | Standalone HIP microbenchmarks |
| [../contrib/system/README.md](../contrib/system/README.md) | System files of the reference machine, as examples |
| [../tests/README.md](../tests/README.md) | How to run the tests, and what they cover |
| [../CHANGELOG.md](../CHANGELOG.md) | Changes by version |
| [../RELEASE_NOTES.md](../RELEASE_NOTES.md) | Notes for the current release |
| [../patches/MODIFIED_FILES.txt](../patches/MODIFIED_FILES.txt) | Files changed against upstream vLLM |
