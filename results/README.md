# Measurement records and tables

| File | Contents |
|---|---|
| `kv-concurrency.jsonl` | Records of `bench/kv-concurrency.py`, one JSON object per line: only those used by the tables, without the per-token timings |
| `runs.json` | Manifest: which records form each run, with the model, code revision, settings and KV cache capacity of the run |
| `all-reduce.json` | All-reduce latency and its end-to-end effect |
| `summary.csv` | All points of all runs in one table: generated |
| `make_tables.py` | Generator, standard library only |
| `local/` | Your own records. Ignored by git |

## Regenerating and checking

```bash
python3 results/make_tables.py            # rewrite the tables in README.md and docs/PERFORMANCE.md, and summary.csv
python3 results/make_tables.py --check    # exit 1 if anything is out of date
python3 results/make_tables.py --list     # names of the tables
```

The tables live between marker comments in the documents:

```
<!-- table:summary -->
...generated...
<!-- /table:summary -->
```

Do not edit a table by hand: the next generation overwrites it.

## How a run selects its records

A run in `runs.json` lists rules in `match`:

```json
{"stage": "pp3-27b-3karty", "contains": "etap=n6", "before": "2026-09-30"}
```

- `stage`: the first word of the record field `etap`.
- `contains` (optional): text that must occur in `etap`.
- `before` (optional): the record field `chas` must sort before this value.

A record counts only if all streams finished without an error. If several records exist for the same prompt length and stream count, the
last one in the file wins: repeats follow fixes.

The settings of a run in `runs.json` (`window`, `kv`, `util`, `max_seqs`, `kv_capacity`, `tree`, `date`) were written by hand from the launch
scripts, logs and notes of the series. In older records `prylad.kv` and `prylad.maxseq` are empty, and `util` is never in a record.

## Points taken from an older run

A run may name another run in `older`. Every point that the run has no records for is taken from that run, and the older run may
name an older one in turn. A taken-over point is marked ` *` in the matrices, the summary and the comparison tables: that number was
measured on an earlier day, with the settings of that run. A point with records of its own is never marked, and a run without `older`
has no marks. An unknown run id or a cycle of `older` stops the generator.

`summary.csv` lists each point under the run that measured it, so a taken-over point appears only under its own run.

## Metrics

| Metric | How it is computed |
|---|---|
| PP | `sum(chyslo.potoky[].prompt_tokens) / max(chyslo.potoky[].chas_do_pershoho_tokena_s)` |
| TG | `chyslo.sukupna_shvydkist_tok_s` |
| TTFT | `max(chyslo.potoky[].chas_do_pershoho_tokena_s)` |

## Record fields

The field names are transliterated Ukrainian, kept so that the records stay comparable with older files.

| Field | Meaning |
|---|---|
| `seriia` | Series name |
| `etap` | Stage: free text, its first word is the stage name |
| `chas` | Time of the record |
| `prylad` | The measuring setup and the server under test |
| `prylad.rushij` | Engine: `vllm` or `llamacpp`, the `--engine` option of the probe |
| `prylad.vahy` | Base name of the weights directory or file |
| `prylad.vikno` | Context window |
| `prylad.maxseq` | Maximum concurrent requests of the server |
| `prylad.spec_n` | Number of draft tokens, 0 without MTP |
| `prylad.kv` | KV cache format |
| `prylad.prapirci` | Server flags |
| `prylad.server_pidnjato` | Whether the probe saw a running server |
| `umovy.potokiv` | Streams |
| `umovy.promt_tokeniv_zadano` | Requested prompt length |
| `umovy.max_tokens` | Requested output length |
| `umovy.blokiv_zagalom` | Blocks of the KV cache, from the server log, 0 if not given |
| `umovy.zerno_bazove` | Base random seed |
| `chyslo.potoky[]` | One entry per stream: tokens, time to the first token, decode speed, error |
| `chyslo.sukupna_shvydkist_tok_s` | Total decode speed in the shared window |
| `chyslo.vikno_spilnoho_dekodu_s` | Length of the shared window |
| `chyslo.prychyna_vidsutnosti_sukupnoi_shvydkosti` | Why there is no total speed: for example, the streams did not overlap |
| `chyslo.kv_cache_usage_perc_pik` | Peak KV cache usage |
| `chyslo.blokiv_pik` | Peak number of blocks in use |
| `chyslo.vytisnen_riznytsia` | Preemptions during the record |
| `chyslo.tryvalist_zagalna_s` | Total duration |
| `chyslo.vram_*` | VRAM use around the record |
| `dokaz.pomylky` | Errors collected by the probe |

## Measuring your own machine

```bash
./serve.sh PRESET
bench/run-matrix.sh PRESET --url http://127.0.0.1:8100 --label my-machine
```

`bench/run-matrix.sh` writes to `results/local/`. Compare with the published records only under the same settings: a different preset, clock
or KV cache format makes the comparison meaningless.
