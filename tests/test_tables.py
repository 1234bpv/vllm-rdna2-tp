"""results/make_tables.py: generated tables in the documents and summary.csv.

The tables between the markers
<!-- table:NAME --> ... <!-- /table:NAME --> in README.md and docs/PERFORMANCE.md, and
summary.csv, are produced by make_tables.py from the measurement records. --check writes
nothing and exits 1 when something is out of date; without arguments the script rewrites
them. The numbers themselves are not recomputed here: that would repeat the script. The
tests check the mechanism: the committed files are current, regeneration reproduces them
byte for byte, a changed cell is detected, and the markers pair up.

Tests that change files do it in a copy of the release tree (release_copy), never in place.

The rules of the generator itself (the "older" runs, the " *" marks, the A/B table, the rows of summary.csv)
are checked on synthetic fixtures written into the sandbox; the committed data is checked against the rules
of the release data (every release run has an older run, the release-tree numbers are valid, no host path).
"""

import csv
import hashlib
import importlib.util
import io
import json
import math
import pathlib
import re
import subprocess
import unittest

from _support import PYTHON, RELEASE, ReleaseTest, write_text

MAKE = pathlib.Path("results") / "make_tables.py"
DOCS = [pathlib.Path("README.md"), pathlib.Path("docs") / "PERFORMANCE.md"]
MARKER = re.compile(r"<!-- (/?)table:([a-z0-9-]+) -->")
KV_RECORDS = RELEASE / "results" / "kv-concurrency.jsonl"
SUMMARY_CSV = RELEASE / "results" / "summary.csv"
PROFILE_POINT = re.compile(r"point=profile")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def profile_record_lines(text):
    """Numbers of the lines that carry the point=profile marker."""
    return [number for number, line in enumerate(text.splitlines(), 1) if PROFILE_POINT.search(line)]


class TablesTest(ReleaseTest):

    def run_make(self, root, args=(), *, timeout=120):
        return self.sb.run([PYTHON, "-I", "-B", str(root / MAKE), *args], timeout=timeout, cwd=root)

    def test_list_names_known_tables(self):
        r = self.sb.run([PYTHON, "-I", "-B", str(RELEASE / MAKE), "--list"], timeout=120)
        self.assertEqual(r.rc, 0, r.describe())
        names = r.out.split()
        for name in ("summary", "all-reduce", "vs-llamacpp", "matrix-q27"):
            self.assertIn(name, names)

    def test_bad_arguments_exit_2(self):
        for args in (["--bogus"], ["--check", "--check"], ["x"]):
            with self.subTest(args=args):
                r = self.sb.run([PYTHON, "-I", "-B", str(RELEASE / MAKE), *args], timeout=120)
                self.assertEqual(r.rc, 2, r.describe())

    def test_markers_pair_up_in_each_document(self):
        for doc in DOCS:
            with self.subTest(doc=doc.as_posix()):
                text = (RELEASE / doc).read_text(encoding="utf-8")
                stack = []
                names_open = []
                for match in MARKER.finditer(text):
                    closing, name = match.group(1) == "/", match.group(2)
                    if not closing:
                        self.assertEqual(stack, [], f"table {name} opens inside another table")
                        stack.append(name)
                        names_open.append(name)
                    else:
                        self.assertEqual(stack.pop() if stack else None, name, f"unmatched /table:{name}")
                self.assertEqual(stack, [], "a table is not closed")
                self.assertEqual(len(names_open), len(set(names_open)), "a table name is used twice")

    def test_committed_tree_is_up_to_date_and_check_writes_nothing(self):
        watched = [RELEASE / doc for doc in DOCS] + [RELEASE / "results" / "summary.csv"]
        before = {p: sha(p) for p in watched}
        r = self.sb.run([PYTHON, "-I", "-B", str(RELEASE / MAKE), "--check"], timeout=120)
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("up to date", r.out)
        self.assertEqual({p: sha(p) for p in watched}, before, "--check changed a file")

    def test_regeneration_reproduces_committed_files(self):
        copy = self.sb.release_copy()
        committed = {copy / doc: sha(copy / doc) for doc in DOCS}
        committed[copy / "results" / "summary.csv"] = sha(copy / "results" / "summary.csv")
        r = self.run_make(copy)
        self.assertEqual(r.rc, 0, r.describe())
        self.assertIn("nothing, already up to date", r.out, r.out)
        self.assertEqual({p: sha(p) for p in committed}, committed, "regeneration changed a committed file")

    def test_changed_cell_in_readme_is_detected_by_check(self):
        copy = self.sb.release_copy()
        readme = copy / "README.md"
        text = readme.read_text(encoding="utf-8")
        lines = text.split("\n")
        start = next(i for i, line in enumerate(lines) if line.startswith("<!-- table:summary -->"))
        for i in range(start + 1, len(lines)):
            if lines[i].startswith("|") and re.search(r"\d", lines[i]) and not lines[i].startswith("| ---"):
                lines[i] = re.sub(r"\d+(\.\d+)?", "999", lines[i], count=1)
                break
        else:
            self.fail("no numeric row found in the summary table")
        tampered = "\n".join(lines)
        readme.write_text(tampered, encoding="utf-8", newline="\n")
        r = self.run_make(copy, ["--check"])
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("out of date", r.out)
        self.assertIn("README.md", r.out)
        self.assertEqual(readme.read_text(encoding="utf-8"), tampered, "--check rewrote the document")
        r = self.run_make(copy)
        self.assertEqual(r.rc, 0, r.describe())
        self.assertEqual(readme.read_text(encoding="utf-8"), text, "the rewrite did not restore the table")

    def test_changed_summary_csv_is_detected_and_restored(self):
        copy = self.sb.release_copy()
        csv_path = copy / "results" / "summary.csv"
        original = csv_path.read_bytes()
        csv_path.write_bytes(original + b"stale,row\n")
        r = self.run_make(copy, ["--check"])
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn("summary.csv", r.out)
        r = self.run_make(copy)
        self.assertEqual(r.rc, 0, r.describe())
        self.assertEqual(csv_path.read_bytes(), original)

    def test_every_listed_table_is_used_in_a_document(self):
        r = self.sb.run([PYTHON, "-I", "-B", str(RELEASE / MAKE), "--check"], timeout=120)
        self.assertNotIn("tables not used in any document", r.out)
        self.assertEqual(r.rc, 0, r.describe())

    def test_unknown_table_marker_is_reported(self):
        copy = self.sb.release_copy()
        readme = copy / "README.md"
        readme.write_text(readme.read_text(encoding="utf-8") +
                          "\n<!-- table:no-such-table -->\nx\n<!-- /table:no-such-table -->\n",
                          encoding="utf-8", newline="\n")
        r = self.run_make(copy, ["--check"])
        self.assertNotEqual(r.rc, 0, r.describe())
        self.assertIn("unknown table in document: no-such-table", r.both)


class MeasurementRecordsTest(ReleaseTest):
    """The committed measurement files: no profile point in the concurrency records, and the reported
    summary value for one configuration. The numbers are read from the files, not recomputed."""

    def test_kv_concurrency_records_parse_and_none_is_a_profile_point(self):
        text = KV_RECORDS.read_text(encoding="utf-8")
        records = [json.loads(line) for line in text.splitlines() if line.strip()]
        self.assertGreater(len(records), 0, "the records file is empty, so the check would prove nothing")
        hits = profile_record_lines(text)
        self.assertEqual(hits, [], f"records with point=profile at lines {hits}")

    def test_profile_detector_finds_a_planted_record(self):
        # The check above is only meaningful if the detector finds the marker when it is present.
        sample = '{"etap": "run point=profile attn=hip"}\n{"etap": "run point=short1 attn=hip"}\n'
        self.assertEqual(profile_record_lines(sample), [1])

    def test_summary_q27_attn_hip_300_tokens_4_streams_has_tg_99_489(self):
        with SUMMARY_CSV.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        match = [row for row in rows if row["run"] == "q27-attn-hip"
                 and row["prompt_tokens"] == "300" and row["streams"] == "4"]
        self.assertEqual(len(match), 1, f"expected one q27-attn-hip row for 300 tokens and 4 streams, found {len(match)}")
        self.assertEqual(match[0]["tg_tok_s"], "99.489")


# ---------------------------------------------------------------------------------------------
# The generator's own logic on synthetic fixtures. A fixture is written into the sandbox: the
# generator reads runs.json, kv-concurrency.jsonl and all-reduce.json from its HERE, so a test
# points HERE at the fixture and calls load() and the table builders in-process. The committed
# files are neither read by these tests nor written.
# ---------------------------------------------------------------------------------------------

GENERATOR = RELEASE / MAKE
RUNS_JSON = RELEASE / "results" / "runs.json"
ALL_REDUCE_JSON = RELEASE / "results" / "all-reduce.json"
# Release data and documents: none of them may hold a host path or a user name.
PUBLISHED = [RELEASE / "README.md", RELEASE / "docs" / "PERFORMANCE.md", KV_RECORDS, RUNS_JSON,
             SUMMARY_CSV, ALL_REDUCE_JSON]
PRIVATE_MARKERS = ("/home/", "/media/", "pa" + "sha", "c:\\")

META = {
    "model": "Test-Model", "checkpoint": "test/checkpoint", "weights": "W4A16, 1 GiB", "engine": "fork",
    "cards": 2, "tp": 2, "pp": 1, "mtp": False, "window": 131072, "kv": "int4", "util": "0.9",
    "max_seqs": 4, "kv_capacity": 1000, "kv_capacity_source": "test", "tree": "t0", "date": "2026-10-01",
}


def load_generator():
    spec = importlib.util.spec_from_file_location("make_tables_under_test", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_spec(run_id, older=None, alt_points=None, **fields):
    """A manifest entry whose records are the ones with the stage run_id (see point_records)."""
    spec = dict(META, id=run_id, match=[{"stage": run_id}], **fields)
    if older is not None:
        spec["older"] = older
    if alt_points is not None:
        spec["alt_points"] = alt_points
    return spec


def point_records(stage, points, failed=False):
    """Records of one run. A point is (prompt tokens per stream, streams, seconds to the first token,
    decode tokens/s). failed=True makes every stream a failed one, as a broken run would be."""
    records = []
    for length, streams, ttft, tg in points:
        threads = [{"nomer": n, "prompt_tokens": length, "chas_do_pershoho_tokena_s": ttft,
                    "pomylka": "stub error" if failed else None} for n in range(streams)]
        records.append({
            "seriia": "kv-concurrency",
            "etap": f"{stage} point=L{length}x{streams}",
            "chas": "2026-10-01 10:00:00",
            "umovy": {"potokiv": streams, "promt_tokeniv_zadano": length, "max_tokens": 200},
            "chyslo": {"potoky": threads, "sukupna_shvydkist_tok_s": tg},
            "dokaz": {"pomylky": ["stub error"] if failed else []},
        })
    return records


def write_fixture(root, name, runs, points, failed=()):
    """Writes the fixture into root/name/results and returns the generator pointed at it.
    points maps a run id to its points; the runs listed in failed get only failed records."""
    results = root / name / "results"
    records = []
    for run in runs:
        records += point_records(run["id"], points.get(run["id"], []), failed=run["id"] in failed)
    write_text(results / "runs.json", json.dumps({"runs": runs}, ensure_ascii=False, indent=1))
    write_text(results / "kv-concurrency.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    write_text(results / "all-reduce.json", json.dumps({"latency": [], "all_collectives": {}, "decode_chain": []}))
    gen = load_generator()
    gen.HERE = results
    return gen


def table_of(text):
    """Header cells and body rows of a generated markdown table. An empty text fails here."""
    lines = [line for line in text.split("\n") if line.startswith("|")]
    rows = [[cell.strip() for cell in line.strip().strip("|").split("|")] for line in lines]
    return rows[0], rows[2:]


def row_of(rows, *prefix):
    """The one row whose first cells equal prefix."""
    found = [row for row in rows if row[:len(prefix)] == list(prefix)]
    if len(found) != 1:
        raise AssertionError(f"expected one row starting with {list(prefix)}, found {len(found)}")
    return found[0]


def named(header, row):
    return dict(zip(header, row))


def own_rule(record, rule):
    """Whether a record belongs to a match rule of runs.json (stage, optional contains and before)."""
    if record["etap"].split(" ")[0] != rule["stage"]:
        return False
    if "contains" in rule and rule["contains"] not in record["etap"]:
        return False
    return "before" not in rule or record["chas"] < rule["before"]


def record_is_valid(record):
    threads = record["chyslo"].get("potoky") or []
    if not threads or (record.get("dokaz") or {}).get("pomylky"):
        return False
    return all(not t.get("pomylka") and t.get("prompt_tokens") and t.get("chas_do_pershoho_tokena_s")
               for t in threads)


def positive_int(value):
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def finite_positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def release_tree_problems(record):
    """Reasons why a release-tree record does not carry valid numbers; empty when it does."""
    problems = []
    umovy = record["umovy"]
    threads = record["chyslo"].get("potoky") or []
    if not positive_int(umovy.get("promt_tokeniv_zadano")):
        problems.append("promt_tokeniv_zadano is not a positive integer")
    if umovy.get("potokiv") != len(threads):
        problems.append(f"potokiv {umovy.get('potokiv')} differs from the {len(threads)} streams recorded")
    if (record.get("dokaz") or {}).get("pomylky"):
        problems.append("dokaz.pomylky is not empty")
    for number, thread in enumerate(threads):
        if thread.get("pomylka") is not None:
            problems.append(f"stream {number} has an error")
        if thread.get("prompt_tokens") != umovy.get("promt_tokeniv_zadano"):
            problems.append(f"stream {number} prompt_tokens differ from the requested length")
        if not finite_positive(thread.get("chas_do_pershoho_tokena_s")):
            problems.append(f"stream {number} time to first token is not a positive finite number")
    if not finite_positive(record["chyslo"].get("sukupna_shvydkist_tok_s")):
        problems.append("sukupna_shvydkist_tok_s is not a positive finite number")
    return problems


class InheritedPointsTest(ReleaseTest):
    """Requirements 1 and 2: a run takes the points it lacks from its older run, its own points stay,
    and a bad link is a SystemExit that names the run."""

    def test_missing_points_come_from_the_older_run_and_are_marked(self):
        gen = write_fixture(self.sb.root, "fx", [run_spec("alpha"), run_spec("beta", older="alpha")],
                            {"alpha": [(300, 1, 0.5, 40.0), (32768, 1, 32.768, 20.0)],
                             "beta": [(300, 4, 1.0, 80.0)]})
        runs = gen.load()[1]
        points = runs["beta"]["points"]
        self.assertEqual(set(points), {(300, 1), (32768, 1), (300, 4)})
        self.assertTrue(points[(300, 1)].older)
        self.assertTrue(points[(32768, 1)].older)
        self.assertFalse(points[(300, 4)].older, "an own point is marked as older")
        self.assertEqual(points[(32768, 1)].tg, 20.0)
        self.assertAlmostEqual(points[(32768, 1)].pp, 1000.0)
        self.assertFalse(any(point.older for point in runs["alpha"]["points"].values()))

    def test_own_point_is_not_replaced_by_the_older_one(self):
        gen = write_fixture(self.sb.root, "fx", [run_spec("alpha"), run_spec("beta", older="alpha")],
                            {"alpha": [(300, 1, 0.5, 99.0)], "beta": [(300, 1, 0.5, 40.0)]})
        point = gen.load()[1]["beta"]["points"][(300, 1)]
        self.assertFalse(point.older)
        self.assertEqual(point.tg, 40.0)

    def test_chain_of_two_older_runs_passes_points_down_nearest_first(self):
        gen = write_fixture(self.sb.root, "fx",
                            [run_spec("alpha"), run_spec("beta", older="alpha"), run_spec("gamma", older="beta")],
                            {"alpha": [(300, 1, 0.5, 99.0), (125000, 1, 100.0, 9.0)],
                             "beta": [(300, 1, 0.5, 30.0), (32768, 1, 32.768, 20.0)],
                             "gamma": [(300, 4, 1.0, 80.0)]})
        points = gen.load()[1]["gamma"]["points"]
        self.assertEqual(set(points), {(300, 1), (32768, 1), (125000, 1), (300, 4)})
        self.assertEqual(points[(300, 1)].tg, 30.0, "taken from alpha, not from the nearer beta")
        self.assertTrue(points[(300, 1)].older)
        self.assertEqual(points[(32768, 1)].tg, 20.0)
        self.assertTrue(points[(32768, 1)].older)
        self.assertTrue(points[(125000, 1)].older)
        self.assertFalse(points[(300, 4)].older)

    def test_run_without_own_records_is_systemexit_even_with_an_older_run(self):
        gen = write_fixture(self.sb.root, "fx", [run_spec("alpha"), run_spec("beta", older="alpha")],
                            {"alpha": [(300, 1, 0.5, 99.0)]})
        with self.assertRaises(SystemExit) as caught:
            gen.load()
        self.assertIn("beta", str(caught.exception))

    def test_run_whose_records_all_failed_is_systemexit_and_does_not_inherit(self):
        gen = write_fixture(self.sb.root, "fx", [run_spec("alpha"), run_spec("beta", older="alpha")],
                            {"alpha": [(300, 1, 0.5, 40.0)], "beta": [(300, 4, 1.0, 80.0)]}, failed=["beta"])
        with self.assertRaises(SystemExit) as caught:
            gen.load()
        self.assertIn("beta", str(caught.exception))

    def test_unknown_older_id_is_systemexit_naming_the_run_and_the_id(self):
        gen = write_fixture(self.sb.root, "fx", [run_spec("alpha"), run_spec("beta", older="gone")],
                            {"alpha": [(300, 1, 0.5, 40.0)], "beta": [(300, 4, 1.0, 80.0)]})
        with self.assertRaises(SystemExit) as caught:
            gen.load()
        message = str(caught.exception)
        self.assertIn("beta", message)
        self.assertIn("gone", message)

    def test_cycle_of_two_older_runs_is_systemexit_naming_both(self):
        gen = write_fixture(self.sb.root, "fx", [run_spec("alpha", older="beta"), run_spec("beta", older="alpha")],
                            {"alpha": [(300, 1, 0.5, 40.0)], "beta": [(300, 4, 1.0, 80.0)]})
        with self.assertRaises(SystemExit) as caught:
            gen.load()
        message = str(caught.exception)
        self.assertIn("alpha", message)
        self.assertIn("beta", message)

    def test_run_that_is_its_own_older_run_is_systemexit_naming_it(self):
        gen = write_fixture(self.sb.root, "fx", [run_spec("alpha", older="alpha")],
                            {"alpha": [(300, 1, 0.5, 40.0)]})
        with self.assertRaises(SystemExit) as caught:
            gen.load()
        self.assertIn("alpha", str(caught.exception))

    def test_cycle_behind_a_run_is_systemexit_naming_the_runs_of_the_cycle(self):
        gen = write_fixture(self.sb.root, "fx",
                            [run_spec("alpha", older="beta"), run_spec("beta", older="gamma"),
                             run_spec("gamma", older="beta")],
                            {"alpha": [(300, 1, 0.5, 40.0)], "beta": [(300, 4, 1.0, 80.0)],
                             "gamma": [(32768, 1, 32.768, 20.0)]})
        with self.assertRaises(SystemExit) as caught:
            gen.load()
        message = str(caught.exception)
        self.assertIn("beta", message)
        self.assertIn("gamma", message)


class RunOrderTest(ReleaseTest):
    """Requirement 3: the order of the runs in runs.json does not change the result."""

    def orders(self):
        base = [run_spec("alpha"), run_spec("beta", older="alpha"), run_spec("gamma", older="beta"),
                run_spec("delta")]
        points = {"alpha": [(300, 1, 0.5, 40.0), (32768, 1, 32.768, 20.0)],
                  "beta": [(300, 4, 1.0, 80.0)],
                  "gamma": [(125000, 2, 50.0, 15.0)],
                  "delta": [(300, 1, 0.4, 41.0)]}
        orders = {"as-written": base, "reversed": list(reversed(base)), "rotated": base[2:] + base[:2]}
        return {name: write_fixture(self.sb.root, name, order, points) for name, order in orders.items()}

    def test_points_of_every_run_do_not_depend_on_the_order_in_runs_json(self):
        seen = {}
        for name, gen in self.orders().items():
            seen[name] = {rid: sorted((key, p.older, round(p.pp, 6), p.tg, round(p.ttft, 6))
                                      for key, p in run["points"].items())
                          for rid, run in gen.load()[1].items()}
        for name in seen:
            with self.subTest(order=name):
                self.assertEqual(seen[name], seen["as-written"])

    def test_matrix_and_comparison_do_not_depend_on_the_order_in_runs_json(self):
        seen = {}
        for name, gen in self.orders().items():
            runs = gen.load()[1]
            seen[name] = (gen.matrix(runs, ["alpha", "beta", "gamma", "delta"], ["a", "b", "c", "d"]),
                          gen.comparison(runs, "gamma", "alpha", [(300, 1), (32768, 1), (300, 4), (125000, 2)]))
        for name in seen:
            with self.subTest(order=name):
                self.assertEqual(seen[name], seen["as-written"])

    def test_summary_csv_rows_as_a_set_do_not_depend_on_the_order_in_runs_json(self):
        seen = {}
        for name, gen in self.orders().items():
            rows = list(csv.reader(io.StringIO(gen.render_csv(gen.load()[1]))))
            seen[name] = (rows[0], set(map(tuple, rows[1:])))
        for name in seen:
            with self.subTest(order=name):
                self.assertEqual(seen[name], seen["as-written"])

    def test_summary_csv_text_is_identical_for_any_order_in_runs_json(self):
        seen = {name: gen.render_csv(gen.load()[1]) for name, gen in self.orders().items()}
        for name in seen:
            with self.subTest(order=name):
                self.assertEqual(seen[name], seen["as-written"], "row order follows runs.json")


class MarkedCellsTest(ReleaseTest):
    """Requirement 4 in the matrix and the comparison: a cell taken from an older run ends in ' *';
    an own cell and an empty cell do not."""

    def setUp(self):
        super().setUp()
        self.gen = write_fixture(self.sb.root, "fx", [run_spec("base"), run_spec("rel", older="base")],
                                 {"base": [(32768, 1, 32.768, 20.0), (300, 1, 0.5, 40.0)],
                                  "rel": [(300, 4, 1.0, 80.0)]})
        self.runs = self.gen.load()[1]
        self.header, self.rows = table_of(self.gen.matrix(self.runs, ["base", "rel"], ["Base", "Rel"]))

    def test_matrix_cell_taken_from_the_older_run_ends_in_star(self):
        self.assertEqual(row_of(self.rows, "32k × 1")[2], "1000 / 20.0 *")

    def test_matrix_own_cell_has_no_star(self):
        self.assertEqual(row_of(self.rows, "32k × 1")[1], "1000 / 20.0")

    def test_matrix_empty_cell_is_a_plain_dash(self):
        self.assertEqual(row_of(self.rows, "300 × 4")[1], "—")

    def test_comparison_cells_taken_from_the_older_run_end_in_star(self):
        row = self.gen.comparison(self.runs, "rel", "base", [(32768, 1)])[0]
        self.assertEqual(row[1:4], ["1000 *", "20.0 *", "33 *"])

    def test_comparison_own_cells_have_no_star(self):
        row = self.gen.comparison(self.runs, "rel", "base", [(32768, 1)])[0]
        self.assertEqual(row[4:7], ["1000", "20.0", "33"])


class SummaryTableMarkTest(ReleaseTest):
    """Requirement 4 in the summary table: the cell of an older point ends in ' *', with the alternative
    length too ('2318 (8k) *'); an own cell and an empty cell do not."""

    def setUp(self):
        super().setUp()
        gen = load_generator()
        specs = {rid: run_spec(rid, model=f"M {rid}") for rid in gen.SUMMARY_ROWS}
        specs["rel-q27"] = run_spec("rel-q27", older="q27", alt_points={"32768": 8192}, model="M rel-q27")
        specs["rel-q35"] = run_spec("rel-q35", older="q35", model="M rel-q35")
        specs["rel-q9"] = run_spec("rel-q9", alt_points={"32768": 8192}, model="M rel-q9")
        specs["q27"] = run_spec("q27")
        specs["q35"] = run_spec("q35")
        points = {rid: [(300, 1, 0.5, 50.0)] for rid in gen.SUMMARY_ROWS}
        points["q27"] = [(8192, 1, 8192 / 2318, 30.0)]
        points["rel-q27"] = [(300, 1, 0.5, 45.0)]
        points["rel-q9"] = [(8192, 1, 8192 / 2318, 31.0)]
        points["q35"] = [(125000, 1, 100.0, 9.0)]
        points["rel-q35"] = [(300, 1, 0.5, 50.0)]
        self.gen = write_fixture(self.sb.root, "fx", list(specs.values()), points)
        self.header, self.rows = table_of(self.gen.t_summary(self.gen.load()[1]))

    def cells(self, run_id):
        return named(self.header, row_of(self.rows, f"M {run_id}"))

    def test_inherited_point_at_the_alternative_length_ends_in_star_after_the_length(self):
        self.assertEqual(self.cells("rel-q27")["PP 32k"], "2318 (8k) *")

    def test_alternative_length_marks_tg_the_same_way(self):
        self.assertEqual(self.cells("rel-q27")["TG 32k"], "30.0 (8k) *")

    def test_inherited_point_without_alternative_ends_in_star(self):
        self.assertEqual(self.cells("rel-q35")["PP 125k"], "1250 *")

    def test_own_point_at_the_alternative_length_has_no_star(self):
        self.assertEqual(self.cells("rel-q9")["PP 32k"], "2318 (8k)")

    def test_empty_cell_is_a_plain_dash(self):
        self.assertEqual(self.cells("rel-q9")["PP 125k"], "—")


class PipelineTableMarkTest(ReleaseTest):
    """Requirement 4 in the pipeline table: an inherited point ends in ' *'; an empty cell does not."""

    def setUp(self):
        super().setUp()
        self.gen = write_fixture(
            self.sb.root, "fx",
            [run_spec("q27"), run_spec("rel-q27", older="q27"), run_spec("rel-pp3-q27"),
             run_spec("gemma"), run_spec("pp3-gemma"), run_spec("gemma-streams")],
            {"q27": [(32768, 1, 32.768, 20.0)], "rel-q27": [(300, 1, 0.5, 45.0)],
             "rel-pp3-q27": [(300, 1, 0.5, 30.0)], "gemma": [(250000, 1, 250.0, 25.0)],
             "pp3-gemma": [(250000, 1, 125.0, 15.0)], "gemma-streams": [(125000, 4, 100.0, 12.0)]})
        self.header, self.rows = table_of(self.gen.t_vs_pp3(self.gen.load()[1]))

    def test_inherited_pp_of_the_two_card_run_ends_in_star(self):
        row = named(self.header, row_of(self.rows, "Qwen3.8-27B", "32k × 1"))
        self.assertEqual(row["TP=2, 2 cards: PP"], "1000 *")

    def test_inherited_tg_of_the_two_card_run_ends_in_star(self):
        row = named(self.header, row_of(self.rows, "Qwen3.8-27B", "32k × 1"))
        self.assertEqual(row["TP=2, 2 cards: TG"], "20.0 *")

    def test_empty_pipeline_cell_is_a_plain_dash(self):
        row = named(self.header, row_of(self.rows, "Qwen3.8-27B", "32k × 1"))
        self.assertEqual(row["Pipeline, 3 cards: PP"], "—")


class ShortPromptMatrixTest(ReleaseTest):
    """Requirement 5 and the 1000-token boundary: PP is a dash for a prompt shorter than 1000 tokens
    with more than one stream; the value is shown otherwise."""

    def setUp(self):
        super().setUp()
        self.gen = write_fixture(self.sb.root, "fx", [run_spec("edge"), run_spec("rel", older="edge")],
                                 {"edge": [(300, 1, 0.5, 40.0), (300, 4, 1.0, 80.0),
                                           (999, 2, 2.0, 50.0), (1000, 2, 2.0, 50.0)],
                                  "rel": [(32768, 1, 32.768, 20.0)]})
        self.runs = self.gen.load()[1]

    def row(self, run_id, length, streams):
        _, rows = table_of(self.gen.matrix(self.runs, [run_id], ["Cell"]))
        return row_of(rows, self.gen.fmt_point(length, streams))[1]

    def test_short_prompt_with_several_streams_shows_tg_only(self):
        self.assertEqual(self.row("edge", 300, 4), "— / 80.0")

    def test_short_prompt_with_several_streams_from_the_older_run_ends_in_star(self):
        self.assertEqual(self.row("rel", 300, 4), "— / 80.0 *")

    def test_short_prompt_with_one_stream_shows_pp(self):
        self.assertEqual(self.row("edge", 300, 1), "600 / 40.0")

    def test_999_tokens_with_several_streams_shows_tg_only(self):
        self.assertEqual(self.row("edge", 999, 2), "— / 50.0")

    def test_1000_tokens_with_several_streams_shows_pp(self):
        self.assertEqual(self.row("edge", 1000, 2), "1000 / 50.0")


AB_ON = ["ab-on-1", "ab-on-2", "ab-on-3"]
AB_OFF = ["ab-off-1", "ab-off-2", "ab-off-3"]


def ab_points(pp_long, tg_short, tg_long):
    """One repeat of an A/B pair: a 300-token point and a 32k point with the given PP."""
    return [(300, 1, 0.5, tg_short), (32768, 1, 32768 / pp_long, tg_long)]


class AllReduceAbTest(ReleaseTest):
    """Requirement 6: the A/B table averages the repeats of each side; the gain comes from the exact means;
    a missing value is a SystemExit."""

    ON = [ab_points(1000, 42.0, 10.5), ab_points(1200, 43.5, 11.2), ab_points(1500, 44.0, 11.3)]
    OFF = [ab_points(900, 40.0, 9.0), ab_points(1000, 41.0, 10.0), ab_points(1100, 43.0, 11.12)]

    def ab_table(self, on_repeats, off_repeats, name="ab"):
        ids = AB_ON + AB_OFF
        gen = write_fixture(self.sb.root, name, [run_spec(rid) for rid in ids],
                            dict(zip(ids, on_repeats + off_repeats)))
        gen.ALL_REDUCE_AB = [("Model M", AB_ON, AB_OFF, [(300, 1), (32768, 1)])]
        return table_of(gen.t_all_reduce_ab(gen.load()[1]))

    def test_tg_columns_show_the_mean_over_the_repeats(self):
        header, rows = self.ab_table(self.ON, self.OFF)
        long = named(header, row_of(rows, "Model M", "32k × 1"))
        short = named(header, row_of(rows, "Model M", "300 × 1"))
        self.assertEqual((long["TG, RCCL"], long["TG, this fork"]), ("10.0", "11.0"))
        self.assertEqual((short["TG, RCCL"], short["TG, this fork"]), ("41.3", "43.2"))

    def test_tg_gain_comes_from_the_exact_means_not_the_shown_ones(self):
        header, rows = self.ab_table(self.ON, self.OFF)
        self.assertEqual(named(header, row_of(rows, "Model M", "32k × 1"))["TG gain"], "+9.6 %")
        self.assertEqual(named(header, row_of(rows, "Model M", "300 × 1"))["TG gain"], "+4.4 %")

    def test_pp_columns_show_the_mean_for_long_prompts_and_a_dash_for_short(self):
        header, rows = self.ab_table(self.ON, self.OFF)
        long = named(header, row_of(rows, "Model M", "32k × 1"))
        short = named(header, row_of(rows, "Model M", "300 × 1"))
        self.assertEqual((long["PP, RCCL"], long["PP, this fork"]), ("1000", "1233"))
        self.assertEqual((short["PP, RCCL"], short["PP, this fork"]), ("—", "—"))

    def test_runs_column_counts_the_repeats_of_each_side(self):
        header, rows = self.ab_table(self.ON, self.OFF)
        self.assertEqual(named(header, row_of(rows, "Model M", "32k × 1"))["Runs"], "3 / 3")

    def test_negative_gain_is_printed_with_a_minus_sign(self):
        on = [ab_points(1000, 10.0, 9.0), ab_points(1000, 10.0, 10.0), ab_points(1000, 10.0, 10.5)]
        off = [ab_points(1000, 10.0, 10.0), ab_points(1000, 10.0, 11.0), ab_points(1000, 10.0, 12.0)]
        header, rows = self.ab_table(on, off)
        self.assertEqual(named(header, row_of(rows, "Model M", "32k × 1"))["TG gain"], "-10.6 %")

    def test_missing_tg_in_one_repeat_is_systemexit_naming_the_run(self):
        on = [self.ON[0], ab_points(1200, 43.5, None), self.ON[2]]
        with self.assertRaises(SystemExit) as caught:
            self.ab_table(on, self.OFF)
        self.assertIn("ab-on-2", str(caught.exception))
        self.assertIn("tg", str(caught.exception))

    def test_missing_point_in_one_repeat_is_systemexit_naming_the_run(self):
        on = [self.ON[0], [(300, 1, 0.5, 43.5)], self.ON[2]]
        with self.assertRaises(SystemExit) as caught:
            self.ab_table(on, self.OFF)
        self.assertIn("ab-on-2", str(caught.exception))


class SummaryCsvFixtureTest(ReleaseTest):
    """Requirement 7 on a fixture: a run lists only the points it has its own records for, once each."""

    def test_each_run_lists_only_its_own_points(self):
        gen = write_fixture(self.sb.root, "fx",
                            [run_spec("alpha"), run_spec("beta", older="alpha"), run_spec("gamma", older="beta")],
                            {"alpha": [(300, 1, 0.5, 40.0), (32768, 1, 32.768, 20.0)],
                             "beta": [(300, 4, 1.0, 80.0)],
                             "gamma": [(300, 1, 0.5, 35.0)]})
        rows = list(csv.DictReader(io.StringIO(gen.render_csv(gen.load()[1]))))
        listed = sorted((row["run"], int(row["prompt_tokens"]), int(row["streams"])) for row in rows)
        self.assertEqual(listed, [("alpha", 300, 1), ("alpha", 32768, 1), ("beta", 300, 4), ("gamma", 300, 1)])

    def test_own_point_is_listed_with_its_own_numbers_not_the_older_ones(self):
        gen = write_fixture(self.sb.root, "fx",
                            [run_spec("alpha"), run_spec("gamma", older="alpha")],
                            {"alpha": [(300, 1, 0.5, 40.0)], "gamma": [(300, 1, 0.5, 35.0)]})
        rows = list(csv.DictReader(io.StringIO(gen.render_csv(gen.load()[1]))))
        gamma = [row for row in rows if row["run"] == "gamma"]
        self.assertEqual(len(gamma), 1)
        self.assertEqual(gamma[0]["tg_tok_s"], "35.000")


def older_problems(runs):
    """Release runs without an older run, or with an older run that is not in runs.json."""
    ids = {run["id"] for run in runs}
    problems = []
    for run in runs:
        if not run["id"].startswith("rel-"):
            continue
        if "older" not in run:
            problems.append(f"{run['id']} has no older run")
        elif run["older"] not in ids:
            problems.append(f"{run['id']}: older run {run['older']} is not in runs.json")
    return problems


def release_tree_problem_lines(records):
    """Number of release-tree records, and one line for each of them that carries invalid numbers."""
    release = [r for r in records if r["etap"].split(" ")[0] == "release-tree"]
    lines = []
    for record in release:
        problems = release_tree_problems(record)
        if problems:
            lines.append(f"{record['etap'].split(chr(10))[0]}: {'; '.join(problems)}")
    return len(release), lines


def own_points(runs, records):
    """(run id, prompt length, streams) of every point with a valid own record under the match rules."""
    keys = set()
    for run in runs:
        for record in records:
            if any(own_rule(record, rule) for rule in run["match"]) and record_is_valid(record):
                umovy = record["umovy"]
                keys.add((run["id"], umovy["promt_tokeniv_zadano"], umovy["potokiv"]))
    return keys


def csv_problems(runs, records, csv_rows):
    """Differences between the rows of summary.csv and the own points of the runs."""
    listed = [(row["run"], int(row["prompt_tokens"]), int(row["streams"])) for row in csv_rows]
    problems = [f"listed twice: {point}" for point in sorted({p for p in listed if listed.count(p) > 1})]
    expected = own_points(runs, records)
    problems += [f"listed without its own record: {point}" for point in sorted(set(listed) - expected)]
    problems += [f"own point not listed: {point}" for point in sorted(expected - set(listed))]
    return problems


def private_hits(name, text):
    hits = []
    for number, line in enumerate(text.splitlines(), 1):
        for marker in PRIVATE_MARKERS:
            if marker in line.lower():
                hits.append(f"{name}:{number}: {marker}")
    return hits


class CommittedDataTest(ReleaseTest):
    """Requirement 8 on the committed data. No count is fixed here: the records keep growing. Each check is
    also shown to flag a planted defect, so that a clean result on the committed data means something."""

    def committed_runs(self):
        return json.loads(RUNS_JSON.read_text(encoding="utf-8"))["runs"]

    def committed_records(self):
        return [json.loads(line) for line in KV_RECORDS.read_text(encoding="utf-8").splitlines() if line.strip()]

    def committed_csv_rows(self):
        with SUMMARY_CSV.open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))

    def test_every_rel_run_names_an_older_run_that_exists(self):
        runs = self.committed_runs()
        self.assertGreater(sum(run["id"].startswith("rel-") for run in runs), 0,
                           "no rel-* run in runs.json, so the check proves nothing")
        self.assertEqual(older_problems(runs), [])

    def test_older_check_flags_a_release_run_without_an_existing_older_run(self):
        runs = [run_spec("rel-a", older="missing"), run_spec("rel-b"), run_spec("base")]
        self.assertEqual(older_problems(runs),
                         ["rel-a: older run missing is not in runs.json", "rel-b has no older run"])

    def test_every_release_tree_record_carries_valid_numbers(self):
        total, bad = release_tree_problem_lines(self.committed_records())
        self.assertGreater(total, 0, "no release-tree record, so the check proves nothing")
        self.assertEqual(bad, [], f"{len(bad)} of {total} release-tree records have invalid numbers")

    def test_release_tree_check_flags_a_planted_record_with_invalid_numbers(self):
        good = point_records("release-tree q27", [(300, 1, 0.5, 40.0)])[0]
        broken = json.loads(json.dumps(good))
        broken["chyslo"]["sukupna_shvydkist_tok_s"] = None
        broken["chyslo"]["potoky"][0]["chas_do_pershoho_tokena_s"] = 0
        total, bad = release_tree_problem_lines([good, broken])
        self.assertEqual(total, 2)
        self.assertEqual(len(bad), 1, bad)
        self.assertIn("sukupna_shvydkist_tok_s", bad[0])
        self.assertIn("time to first token", bad[0])

    def test_summary_csv_lists_exactly_the_own_points_of_each_run(self):
        rows = self.committed_csv_rows()
        self.assertGreater(len(rows), 0, "summary.csv has no rows")
        self.assertEqual(csv_problems(self.committed_runs(), self.committed_records(), rows), [])

    def test_csv_check_flags_planted_duplicate_extra_and_missing_rows(self):
        runs, records, rows = self.committed_runs(), self.committed_records(), self.committed_csv_rows()
        self.assertTrue(any(p.startswith("listed twice") for p in csv_problems(runs, records, rows + [rows[0]])))
        planted = {"run": "rel-q27", "prompt_tokens": "999999", "streams": "9"}
        self.assertTrue(any(p.startswith("listed without its own record")
                            for p in csv_problems(runs, records, rows + [planted])))
        self.assertTrue(any(p.startswith("own point not listed") for p in csv_problems(runs, records, rows[:-1])))

    def test_no_host_path_or_user_name_in_the_published_data(self):
        hits = []
        for path in PUBLISHED:
            hits += private_hits(path.relative_to(RELEASE).as_posix(), path.read_text(encoding="utf-8"))
        self.assertEqual(hits, [])

    def test_host_path_check_flags_planted_paths(self):
        # The sample is assembled from pieces, so that no private path appears verbatim in this file.
        text = "ok line\nsee " + "/" + "home" + "/relt/x\nC:" + "\\" + "Users\\x\nuser Pa" + "sha here\n"
        self.assertEqual(private_hits("sample", text),
                         ["sample:2: /home/", "sample:3: c:\\", "sample:4: pa" + "sha"])


class LineEndingTest(ReleaseTest):
    """Release files are LF (.gitattributes). A CRLF copy of a generated file is not the file the generator
    writes, and --check has to report it, as the byte-for-byte promise of this module says."""

    def check_crlf(self, relative, name):
        copy = self.sb.release_copy()
        path = copy / relative
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
        r = self.sb.run([PYTHON, "-I", "-B", str(copy / MAKE), "--check"], timeout=120, cwd=copy)
        self.assertEqual(r.rc, 1, r.describe())
        self.assertIn(name, r.out, r.describe())

    def test_check_reports_a_crlf_summary_csv_as_out_of_date(self):
        self.check_crlf(pathlib.Path("results") / "summary.csv", "summary.csv")

    def test_check_reports_a_crlf_readme_as_out_of_date(self):
        self.check_crlf(pathlib.Path("README.md"), "README.md")


class UnmarkedTablesTest(ReleaseTest):
    """Latent, not in the committed data today. The docstring of make_tables.py promises that an older point is
    marked in the tables. t_repeats and t_components (tested here), and t_kv, t_tp_scaling and t_all_reduce_ab
    (the same code path, not tested here), print values from runs.json without the mark. The committed tables
    take no older point there, so no committed cell is wrong now."""

    def setUp(self):
        super().setUp()
        self.gen = write_fixture(self.sb.root, "fx", [run_spec("base"), run_spec("rel", older="base")],
                                 {"base": [(32768, 1, 32.768, 20.0)], "rel": [(300, 1, 0.5, 40.0)]})
        self.runs = self.gen.load()[1]

    def test_repeats_table_marks_a_value_taken_from_an_older_run(self):
        self.gen.REPEATS = [("Pair", "base", "rel", [(32768, 1)])]
        header, rows = table_of(self.gen.t_repeats(self.runs))
        self.assertIn("*", named(header, row_of(rows, "Pair", "32k × 1"))["TG, run A / run B"])

    def test_components_table_marks_a_value_taken_from_an_older_run(self):
        self.gen.COMPONENTS = [("Pair", "base", "rel", "tg", [(32768, 1)])]
        header, rows = table_of(self.gen.t_components(self.runs))
        self.assertIn("*", named(header, row_of(rows, "Pair", "TG", "32k × 1"))["With"])
