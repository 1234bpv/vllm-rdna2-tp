"""serve.sh: arguments, --print-command, presets, MTP head, --list and --help.

The values of each preset are checked against the PRESETS table below. Where the tests have to choose a
value, the choice is named in a comment; such choices are interpretations and are listed in tests/README.md.
"""

import json
import pathlib

from _support import ReleaseTest, flag_positions, flag_value, msys, parse_printed_command

REASON_PHRASE = ("Okay, I've spent enough time on this. Let me wrap up my reasoning and "
                 "give the answer.")
CAPTURE_16 = [1, 2, 3, 4, 6, 8, 9, 12, 15, 16, 18, 21, 24, 32, 40, 48]
CAPTURE_11 = [1, 2, 3, 4, 6, 8, 9, 12, 16, 18, 24]
W4A8_ENV = {"VLLM_ROCM_W4A8": "256", "VLLM_ROCM_W4A8_TILE": "8", "VLLM_ROCM_LMHEAD_INT8": "1",
            "VLLM_ROCM_EMBED_UVA": "1", "VLLM_ROCM_MIN_P_FLOOR": "0.05",
            "VLLM_ROCM_MAX_TEMPERATURE": "0"}
MOE_ENV = {"VLLM_ROCM_MOE_RDNA2": "1", "VLLM_ROCM_MAX_TEMPERATURE": "0"}

# One row per preset with the values that serve.sh must print for it.
#   vision_default: vision is on by default; vision_kwargs: vision, when on, passes --mm-processor-kwargs.
PRESETS = {
    "qwen3.8-27b": dict(hf="philbert440/Qwen3.8-27B-W4A16-AWQ", served="qwen3.8-27b", ctx="262144",
                        util="0.92", maxseq="8", mtp_default=False, vision_default=False,
                        vision_kwargs=True, parser="qwen3_xml", reasoning="qwen3",
                        reason_end=True, capture=CAPTURE_16, env=W4A8_ENV),
    "qwen3.5-9b": dict(hf="atalsharma/Qwen3.5-9B-AWQ", served="qwen3.5-9b", ctx="262144",
                       util="0.92", maxseq="8", mtp_default=False, vision_default=False,
                       vision_kwargs=True, parser="qwen3_xml", reasoning="qwen3",
                       reason_end=True, capture=CAPTURE_16, env=W4A8_ENV),
    "qwen3.5-35b-a3b": dict(hf="cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit", served="qwen3.5-35b-a3b",
                            ctx="131072", util="0.97", maxseq="4", mtp_default=False,
                            vision_default=True, vision_kwargs=True, parser="qwen3_xml",
                            reasoning="qwen3", reason_end=True, capture=CAPTURE_16, env=MOE_ENV),
    "qwen3.6-35b-a3b": dict(local="qwen3.6-35b-a3b-ct-int4", served="qwen3.6-35b-a3b",
                            ctx="262144", util="0.97", maxseq="8", mtp_default=True,
                            vision_default=True, vision_kwargs=True, parser="qwen3_xml",
                            reasoning="qwen3", reason_end=True, capture=CAPTURE_11,
                            env={"VLLM_ROCM_MOE_RDNA2": "1", "VLLM_ROCM_W8A16": "1",
                                 "VLLM_ROCM_MAX_TEMPERATURE": "0"}),
    "gemma-4-26b-a4b": dict(hf="cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4", served="gemma-4-26b-a4b",
                            ctx="262144", util="0.97", maxseq="16", mtp_default=False,
                            vision_default=True, vision_kwargs=False, parser="gemma4",
                            reasoning="gemma4", reason_end=False, capture=None, env=MOE_ENV),
}


class ServeTestBase(ReleaseTest):

    def preset_model(self, name):
        """Creates the converted weights directory for a CONVERT preset; returns the model value."""
        spec = PRESETS[name]
        if "local" in spec:
            path = self.sb.rdh / "models" / spec["local"]
            path.mkdir(parents=True, exist_ok=True)
            return path
        return spec["hf"]

    def print_cmd(self, args, *, env=None, timeout=60):
        # --print-command must precede "--": everything after it is passed to vllm.
        args = list(args)
        cut = args.index("--") if "--" in args else len(args)
        args = args[:cut] + ["--print-command"] + args[cut:]
        return self.sb.bash("serve.sh", args, timeout=timeout, env=env)

    def parsed(self, args, *, env=None):
        r = self.print_cmd(args, env=env)
        self.assertEqual(r.rc, 0, r.describe())
        return parse_printed_command(r.out)


class PathOrderTest(ServeTestBase):
    """The PATH line of --print-command: the venv bin, the ROCm bin, then a reference to the caller's PATH.

    The printed line keeps "$PATH" unexpanded, because this output goes into bug reports, so no entry of
    the caller's PATH may appear in it. The PATH that the server process gets is expanded; the launched
    processes are checked in test_serve_launch.py.
    """

    def venv_rocm(self):
        return f"{msys(self.sb.rdh / 'venv' / 'bin')}:{msys(self.sb.rocm / 'bin')}"

    def test_print_command_path_is_venv_then_rocm_then_the_previous_path(self):
        for name in PRESETS:
            with self.subTest(preset=name):
                self.preset_model(name)
                r = self.print_cmd([name])
                self.assertEqual(r.rc, 0, r.describe())
                self.assertIn(f'PATH={self.venv_rocm()}:"$PATH" \\', r.out)
                env, _ = parse_printed_command(r.out)
                self.assertEqual(env["PATH"], f"{self.venv_rocm()}:$PATH")

    def test_caller_path_entries_never_appear_in_the_printed_command(self):
        marker = "/opt/rt-custom/bin"
        base = self.sb.env()["PATH"]
        r = self.print_cmd(["qwen3.8-27b"], env={"PATH": marker + ":" + base})
        self.assertEqual(r.rc, 0, r.describe())
        self.assertNotIn(marker, r.out, "an entry of the caller's PATH is in the printed command")
        self.assertNotIn(base, r.out, "the caller's PATH is printed expanded")
        self.assertIn(f'PATH={self.venv_rocm()}:"$PATH" \\', r.out)


class PresetValuesTest(ServeTestBase):

    def test_print_command_needs_no_install_marker(self):
        # --print-command does not check the installation.
        self.assertFalse(self.sb.marker().exists())
        r = self.print_cmd(["qwen3.8-27b"])
        self.assertEqual(r.rc, 0, r.describe())

    def test_every_preset_matches_spec_table(self):
        for name, spec in PRESETS.items():
            with self.subTest(preset=name):
                model = self.preset_model(name)
                env, argv = self.parsed([name])
                self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0,1")
                self.assertEqual(env["ROCM_PATH"], msys(self.sb.rocm))
                self.assertEqual(env["PYTORCH_ROCM_ARCH"], "gfx1030")
                self.assertEqual(env["SAFETENSORS_FAST_GPU"], "1")
                self.assertEqual(env["OMP_NUM_THREADS"], "4")
                for key, value in spec["env"].items():
                    self.assertEqual(env.get(key), value, f"PRESET_ENV {key}")
                self.assertTrue(argv[0].endswith("venv/bin/vllm"), argv[0])
                self.assertEqual(argv[1], "serve")
                if isinstance(model, pathlib.Path):
                    self.assertTrue(argv[2].endswith("models/" + spec["local"]), argv[2])
                else:
                    self.assertEqual(argv[2], model)
                self.assertEqual(flag_value(argv, "--served-model-name"), spec["served"])
                self.assertEqual(flag_value(argv, "--max-model-len"), spec["ctx"])
                self.assertEqual(flag_value(argv, "--kv-cache-dtype"), "int4_per_token_head")
                self.assertEqual(flag_value(argv, "--gpu-memory-utilization"), spec["util"])
                self.assertEqual(flag_value(argv, "--max-num-seqs"), spec["maxseq"])
                self.assertEqual(flag_value(argv, "--max-num-batched-tokens"), "8192")
                self.assertEqual(flag_value(argv, "--long-prefill-token-threshold"), "2048")
                self.assertEqual(flag_value(argv, "--attention-backend"), "TRITON_ATTN")
                self.assertEqual(flag_value(argv, "--host"), "127.0.0.1")
                self.assertEqual(flag_value(argv, "--port"), "8100")
                self.assertEqual(flag_value(argv, "--shutdown-timeout"), "60")
                self.assertEqual(flag_value(argv, "--dtype"), "float16")
                self.assertIn("--disable-custom-all-reduce", argv)
                self.assertEqual(flag_value(argv, "--tool-call-parser"), spec["parser"])
                self.assertIn("--enable-auto-tool-choice", argv, "TOOLS default is 1")
                self.assertEqual(flag_value(argv, "--reasoning-parser"), spec["reasoning"])
                if spec["reason_end"]:
                    cfg = json.loads(flag_value(argv, "--reasoning-config"))
                    self.assertEqual(cfg["reasoning_end_str"], REASON_PHRASE)
                else:
                    self.assertNotIn("--reasoning-config", argv)
                self.assertEqual(flag_value(argv, "--pipeline-parallel-size"), "1")
                self.assertEqual(flag_value(argv, "--tensor-parallel-size"), "2")
                self.assertEqual(flag_positions(argv, "--pipeline-parallel-size")[-1] + 2,
                                 flag_positions(argv, "--tensor-parallel-size")[-1])

    def test_vision_default_and_kwargs_follow_table(self):
        for name, spec in PRESETS.items():
            with self.subTest(preset=name):
                self.preset_model(name)
                _, argv = self.parsed([name])
                if spec["vision_default"] and spec["vision_kwargs"]:
                    self.assertEqual(flag_value(argv, "--mm-processor-kwargs"), '{"max_pixels":1048576}')
                    self.assertIsNone(flag_value(argv, "--limit-mm-per-prompt"))
                elif spec["vision_default"]:
                    self.assertNotIn("--mm-processor-kwargs", argv)
                    self.assertIsNone(flag_value(argv, "--limit-mm-per-prompt"))
                else:
                    self.assertNotIn("--mm-processor-kwargs", argv)
                    self.assertEqual(flag_value(argv, "--limit-mm-per-prompt"), '{"image":0,"video":0}')

    def test_mtp_default_follows_preset(self):
        for name, spec in PRESETS.items():
            with self.subTest(preset=name):
                self.preset_model(name)
                env, argv = self.parsed([name])
                if spec["mtp_default"]:
                    self.assertEqual(env["VLLM_ROCM_INT4_DECODE_QLEN_MAX"], "3")
                    self.assertIn("--speculative-config", argv)
                else:
                    self.assertEqual(env["VLLM_ROCM_INT4_DECODE_QLEN_MAX"], "1")
                    self.assertNotIn("--speculative-config", argv)
                    self.assertNotIn("--compilation-config", argv)

    def test_mtp_flags_match_preset_capture_sizes(self):
        # qwen3.5-9b needs its MTP head; it is covered in MtpHeadTest instead.
        for name, spec in PRESETS.items():
            if spec["capture"] is None or name == "qwen3.5-9b":
                continue
            with self.subTest(preset=name):
                self.preset_model(name)
                env, argv = self.parsed([name, "--mtp"])
                self.assertEqual(env["VLLM_ROCM_INT4_DECODE_QLEN_MAX"], "3")
                spec_cfg = json.loads(flag_value(argv, "--speculative-config"))
                self.assertEqual(spec_cfg, {"method": "qwen3_5_mtp", "num_speculative_tokens": 2})
                comp = json.loads(flag_value(argv, "--compilation-config"))
                self.assertEqual(comp, {"cudagraph_capture_sizes": spec["capture"]})

    def test_gemma_has_no_mtp_method_so_mtp_is_code2(self):
        # A preset with an empty MTP method (gemma) gives exit code 2 for --mtp.
        r = self.print_cmd(["gemma-4-26b-a4b", "--mtp"])
        self.assertEqual(r.rc, 2, r.describe())

    def test_no_mtp_switches_off_mtp_on_qwen36(self):
        self.preset_model("qwen3.6-35b-a3b")
        env, argv = self.parsed(["qwen3.6-35b-a3b", "--no-mtp"])
        self.assertEqual(env["VLLM_ROCM_INT4_DECODE_QLEN_MAX"], "1")
        self.assertNotIn("--speculative-config", argv)
        self.assertNotIn("--compilation-config", argv)

    def test_tool_flags_present_by_default_and_removed_by_no_tools(self):
        for name in PRESETS:
            with self.subTest(preset=name):
                self.preset_model(name)
                _, argv = self.parsed([name])
                self.assertIn("--enable-auto-tool-choice", argv)
        _, argv = self.parsed(["qwen3.8-27b", "--no-tools"])
        self.assertNotIn("--enable-auto-tool-choice", argv)
        self.assertNotIn("--tool-call-parser", argv)

    def test_convert_preset_without_weights_is_code2_with_download_hint(self):
        # A CONVERT preset whose weights directory is missing gives exit code 2 with a hint to run download-model.sh.
        r = self.print_cmd(["qwen3.6-35b-a3b"])
        self.assertEqual(r.rc, 2, r.describe())
        self.assertIn("download-model.sh", r.err)

    def test_list_and_help_exit_zero_without_install(self):
        listing = self.sb.bash("serve.sh", ["--list"], timeout=60)
        self.assertEqual(listing.rc, 0, listing.describe())
        for name in PRESETS:
            with self.subTest(preset=name):
                self.assertIn(name, listing.out)
        helptext = self.sb.bash("serve.sh", ["--help"], timeout=60)
        self.assertEqual(helptext.rc, 0, helptext.describe())
        for option in ["--model", "--mtp", "--no-mtp", "--ctx", "--streams", "--kv", "--util",
                       "--host", "--port", "--gpus", "--tools", "--no-tools", "--vision",
                       "--no-vision", "--offload-gb", "--tries", "--start-timeout",
                       "--foreground", "--print-command", "--list", "--help"]:
            with self.subTest(option=option):
                self.assertIn(option, helptext.out)


class OverrideAndOptionTest(ServeTestBase):

    def test_options_override_preset(self):
        r = self.print_cmd(["qwen3.8-27b", "--ctx", "4096", "--streams", "2", "--kv", "int8",
                            "--util", "0.5", "--host", "0.0.0.0", "--port", "9000",
                            "--gpus", "0", "--no-tools", "--no-vision"])
        self.assertEqual(r.rc, 0, r.describe())
        env, argv = parse_printed_command(r.out)
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0")
        self.assertEqual(flag_value(argv, "--tensor-parallel-size"), "1")
        self.assertEqual(flag_value(argv, "--max-model-len"), "4096")
        self.assertEqual(flag_value(argv, "--max-num-seqs"), "2")
        self.assertEqual(flag_value(argv, "--gpu-memory-utilization"), "0.5")
        self.assertEqual(flag_value(argv, "--host"), "0.0.0.0")
        self.assertEqual(flag_value(argv, "--port"), "9000")
        self.assertNotIn("--enable-auto-tool-choice", argv)
        self.assertEqual(flag_value(argv, "--limit-mm-per-prompt"), '{"image":0,"video":0}')

    def test_kv_option_mapping(self):
        # Interpretation: --kv int4|int8 map to the vLLM names *_per_token_head (the preset
        # table gives int4_per_token_head for KV); auto passes through.
        _, argv = self.parsed(["qwen3.8-27b", "--kv", "auto"])
        self.assertEqual(flag_value(argv, "--kv-cache-dtype"), "auto")
        _, argv = self.parsed(["qwen3.8-27b", "--kv", "int4"])
        self.assertEqual(flag_value(argv, "--kv-cache-dtype"), "int4_per_token_head")
        _, argv = self.parsed(["qwen3.8-27b", "--kv", "int8"])
        self.assertEqual(flag_value(argv, "--kv-cache-dtype"), "int8_per_token_head")

    def test_model_option_hf_id_and_directory(self):
        _, argv = self.parsed(["qwen3.8-27b", "--model", "someone/else-model"])
        self.assertEqual(argv[2], "someone/else-model")
        model_dir = self.sb.root / "mymodel"
        model_dir.mkdir()
        _, argv = self.parsed(["qwen3.8-27b", "--model", msys(model_dir)])
        self.assertTrue(argv[2].endswith("mymodel"), argv[2])

    def test_vision_flag_turns_vision_on(self):
        _, argv = self.parsed(["qwen3.8-27b", "--vision"])
        self.assertEqual(flag_value(argv, "--mm-processor-kwargs"), '{"max_pixels":1048576}')
        self.assertIsNone(flag_value(argv, "--limit-mm-per-prompt"))
        _, argv = self.parsed(["gemma-4-26b-a4b", "--no-vision"])
        self.assertEqual(flag_value(argv, "--limit-mm-per-prompt"), '{"image":0,"video":0}')

    def test_offload_flag(self):
        # The offload connector config has no block_size key and carries the size in bytes; the name of the bytes key is not checked.
        _, argv = self.parsed(["qwen3.8-27b", "--offload-gb", "4"])
        cfg = json.loads(flag_value(argv, "--kv-transfer-config"))
        self.assertEqual(cfg["kv_connector"], "OffloadingConnector")
        self.assertNotIn("block_size", json.dumps(cfg))
        self.assertIn(str(4 * 2 ** 30), json.dumps(cfg), "4 GiB is not in the connector config")
        _, argv = self.parsed(["qwen3.8-27b", "--offload-gb", "0"])
        self.assertNotIn("--kv-transfer-config", argv)

    def test_extra_arguments_come_before_parallel_sizes(self):
        _, argv = self.parsed(["qwen3.8-27b", "--", "--max-num-seqs", "3", "--enable-prefix-caching"])
        self.assertIn("--enable-prefix-caching", argv)
        user = flag_positions(argv, "--max-num-seqs")
        self.assertEqual(len(user), 2, argv)
        self.assertEqual(argv[user[-1] + 1], "3", "the user value must be the last one")
        pipeline = argv.index("--pipeline-parallel-size")
        tensor = argv.index("--tensor-parallel-size")
        self.assertLess(user[-1], pipeline)
        self.assertLess(pipeline, tensor)
        self.assertEqual(tensor, len(argv) - 2, "parallel sizes must end the command")

    def test_double_dash_cannot_change_the_parallel_sizes(self):
        # CONFIGURATION.md: serve.sh always puts the parallel sizes last, so the arguments after -- cannot
        # change them. The user's sizes stay in the command, before serve.sh's own pair.
        _, argv = self.parsed(["qwen3.8-27b", "--", "--tensor-parallel-size", "4",
                               "--pipeline-parallel-size", "2"])
        self.assertEqual(argv[-4:], ["--pipeline-parallel-size", "1", "--tensor-parallel-size", "2"], argv[-6:])
        self.assertEqual(flag_value(argv, "--tensor-parallel-size"), "2")
        self.assertEqual(flag_value(argv, "--pipeline-parallel-size"), "1")
        user = flag_positions(argv, "--tensor-parallel-size")
        self.assertEqual(len(user), 2, "the user's value was dropped instead of kept before the pair")
        self.assertEqual(argv[user[0] + 1], "4")

    def test_reasoning_end_from_environment_for_gemma(self):
        # The gemma reasoning phrase is set only when REASONEND is passed explicitly.
        _, argv = self.parsed(["gemma-4-26b-a4b"], env={"REASONEND": "custom phrase"})
        self.assertEqual(json.loads(flag_value(argv, "--reasoning-config"))["reasoning_end_str"],
                         "custom phrase")
        _, argv = self.parsed(["gemma-4-26b-a4b"])
        self.assertNotIn("--reasoning-config", argv)

    def test_empty_reasoning_end_disables_reasoning_config(self):
        _, argv = self.parsed(["qwen3.8-27b"], env={"REASONEND": ""})
        self.assertNotIn("--reasoning-config", argv)

    def test_default_reasoning_phrase_round_trips(self):
        _, argv = self.parsed(["qwen3.8-27b"])
        self.assertEqual(json.loads(flag_value(argv, "--reasoning-config"))["reasoning_end_str"],
                         REASON_PHRASE)

    def test_gpu_list_sets_visible_devices_and_tensor_size(self):
        env, argv = self.parsed(["qwen3.8-27b", "--gpus", "0"])
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0")
        self.assertEqual(flag_value(argv, "--tensor-parallel-size"), "1")
        env, _ = self.parsed(["qwen3.8-27b", "--gpus", "0,1"])
        self.assertEqual(env["HIP_VISIBLE_DEVICES"], "0,1")


class InvalidArgumentTest(ServeTestBase):
    """Exit code 2 for a wrong call or an unknown preset."""

    def test_invalid_arguments_are_code2(self):
        # Interpretation: --tries 0 and --start-timeout 0 are wrong calls (N must be positive).
        bad = [
            ["qwen3.8-27b", "--ctx", "0"], ["qwen3.8-27b", "--ctx", "abc"],
            ["qwen3.8-27b", "--streams", "0"], ["qwen3.8-27b", "--kv", "fp8"],
            ["qwen3.8-27b", "--util", "1.5"], ["qwen3.8-27b", "--util", "x"],
            ["qwen3.8-27b", "--port", "0"], ["qwen3.8-27b", "--port", "70000"],
            ["qwen3.8-27b", "--port", "x"], ["qwen3.8-27b", "--gpus", "1,0"],
            ["qwen3.8-27b", "--gpus", "0,"], ["qwen3.8-27b", "--gpus", "a"],
            ["qwen3.8-27b", "--offload-gb", "-1"], ["qwen3.8-27b", "--tries", "0"],
            ["qwen3.8-27b", "--start-timeout", "0"], ["qwen3.8-27b", "--model"],
            ["qwen3.8-27b", "--ctx"], ["qwen3.8-27b", "--bogus"],
            ["unknown-preset"], ["../etc/x"], ["qwen3.8-27b", "qwen3.5-9b"], [],
        ]
        for args in bad:
            with self.subTest(args=args):
                # Run as given, without --print-command: a wrong call is code 2 before anything else.
                r = self.sb.bash("serve.sh", args, timeout=60)
                self.assertEqual(r.rc, 2, r.describe())

    def test_reversed_gpu_order_is_code2(self):
        # --gpus with numbers that are not in ascending order gives exit code 2.
        r = self.print_cmd(["qwen3.8-27b", "--gpus", "1,0"])
        self.assertEqual(r.rc, 2, r.describe())

    def test_boundary_values_are_accepted(self):
        for args in (["--util", "0.5"], ["--port", "1"], ["--port", "65535"],
                     ["--gpus", "0"], ["--offload-gb", "0"], ["--streams", "1"]):
            with self.subTest(args=args):
                r = self.print_cmd(["qwen3.8-27b"] + args)
                self.assertEqual(r.rc, 0, r.describe())


class MtpHeadTest(ServeTestBase):
    """qwen3.5-9b needs its MTP head file before --mtp is accepted."""

    def test_mtp_on_local_dir_without_head_is_code2_with_add_mtp_hint(self):
        model = self.sb.rdh / "models" / "qwen3.5-9b-nohead"
        model.mkdir(parents=True)
        r = self.print_cmd(["qwen3.5-9b", "--mtp", "--model", msys(model)])
        self.assertEqual(r.rc, 2, r.describe())
        self.assertIn("scripts/add-mtp-head.py", r.err)

    def test_mtp_with_hf_id_is_code2_with_download_hint(self):
        r = self.print_cmd(["qwen3.5-9b", "--mtp"])
        self.assertEqual(r.rc, 2, r.describe())
        self.assertIn("scripts/download-model.sh", r.err)

    def test_mtp_accepts_either_head_file_name(self):
        # --mtp is rejected only when neither model-mtp.safetensors nor model_mtp.safetensors is present.
        for head_name, dir_name in (("model-mtp.safetensors", "head-a"),
                                    ("model_mtp.safetensors", "head-b")):
            with self.subTest(head=head_name):
                model = self.sb.rdh / "models" / dir_name
                model.mkdir(parents=True)
                (model / head_name).write_bytes(b"x")
                env, argv = self.parsed(["qwen3.5-9b", "--mtp", "--model", msys(model)])
                self.assertEqual(env["VLLM_ROCM_INT4_DECODE_QLEN_MAX"], "3")
                comp = json.loads(flag_value(argv, "--compilation-config"))
                self.assertEqual(comp, {"cudagraph_capture_sizes": CAPTURE_16})

    def test_9b_without_mtp_needs_no_head(self):
        r = self.print_cmd(["qwen3.5-9b"])
        self.assertEqual(r.rc, 0, r.describe())
