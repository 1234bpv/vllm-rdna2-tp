# Installation

This guide goes from a clean Linux system to a running server. `README.md` has the [short version](../README.md#quick-start).

**Status for version 0.1.0.** This guide was followed literally on 5 Nov 2026 in a clean Ubuntu 24.04 container on the reference machine: ROCm 7.2.4 (22 GB, about 15 minutes), `install.sh` and `doctor.sh` both returned 0, Qwen3.5-9B was downloaded (7.2 GB in 68 s) and started without `--model` (549 s, a first start), and three matrix points were measured. Installing from zero directly on the machine took 912 s and created a 17 GB virtual environment and 0.8 GB of sources, and the first start of Qwen3.5-9B on that install took 523 s. Removing `libpython3.12-dev` made `doctor.sh` fail and `install.sh` exit with code 3, as documented. No other machine has been tested. Please report what differs on yours.

## 1. Requirements

| Item | Tested | Notes |
|---|---|---|
| GPUs | RX 6800 + RX 6800 XT (gfx1030) | Other gfx1030 cards were not tested. Other RDNA2 chips (gfx1031 and later) are not expected to work with these settings |
| OS | Kubuntu 24.04, Linux 6.8 | Linux only. On another distribution `doctor.sh` warns, and the commands below may need changes |
| ROCm | 7.2.4, without DKMS | Uses the in-kernel `amdgpu` driver |
| Python | 3.12 | 3.10–3.14 are accepted with a warning |
| System packages | `git`, `g++`, `make`, `curl`, `python3.12`, `python3.12-dev`, `python3.12-venv` | |
| Rust | Not needed | The Rust extensions of vLLM are optional, and without a compiler the build skips them. No preset uses them |
| Disk | | See [the table of models](#4-download-a-model). The virtual environment comes on top: PyTorch alone is a 6.2 GB download |
| Host memory | 128 GB | `doctor.sh` warns below 32 GB. The real minimum was not measured |
| Network | | Needed during installation: GitHub, PyPI, `download.pytorch.org`, `repo.radeon.com`, Hugging Face |

The fork does not need P2P access between the cards. See [ARCHITECTURE.md](ARCHITECTURE.md#tensor-parallelism-without-p2p).

## 2. Install ROCm

`install.sh` never uses `sudo`, so this part is yours. These commands were used on the reference machine:

```bash
sudo apt-get update
sudo apt-get install -y git g++ make curl lsof python3.12 python3.12-dev python3.12-venv

curl -LO https://repo.radeon.com/amdgpu-install/7.2.4/ubuntu/noble/amdgpu-install_7.2.4.70204-1_all.deb
sudo apt-get install -y ./amdgpu-install_7.2.4.70204-1_all.deb
sudo apt-get update
sudo amdgpu-install --usecase=rocm --no-dkms -y
sudo usermod -aG render,video "$USER"    # then log out and back in
```

AMD lists gfx1030 only as W6800 and V620, not as RX 6800 or RX 6800 XT. The cards still work with this ROCm version on the reference machine.

## 3. Install vLLM

```bash
git clone https://github.com/1234bpv/vllm-rdna2-tp.git vllm-rdna2-tp
cd vllm-rdna2-tp

timeout 120 scripts/doctor.sh --pre    # check the system; nothing is changed
./install.sh
```

**What `doctor.sh` checks.** Each check prints `PASS`, `WARN` or `FAIL` and a detail. The exit code is 1 if any check fails.

| Check | `FAIL` when | `WARN` when |
|---|---|---|
| `os` | not Linux | not Ubuntu 24.04 |
| `kfd` | `/dev/kfd` is missing or not accessible | |
| `groups` | | the user is in neither `render` nor `video` |
| `rocm` | `$ROCM_PATH/bin/rocminfo` is missing | |
| `rocm-version` | | not 7.2 |
| `hipcc` | `$ROCM_PATH/bin/hipcc` is missing | |
| `gpu-count` | no gfx1030 device | one device, or other gfx103x devices too |
| `python` | no Python 3.10–3.14 | not 3.12 |
| `build-tools` | `git`, `g++` or `make` is missing | |
| `python-dev` | `Python.h` is missing | |
| `ram` | | below 32 GiB |
| `shm` | | below 8 GiB free in `/dev/shm` |
| `disk` | | below 40 GiB free in the install location |
| `venv`, `ninja`, `vllm`, `p2p` | Skipped with `--pre`. After installation: the venv or its `ninja` is missing, or vLLM does not import. `p2p` only reports | |

**What `install.sh` does.** It prints every command and writes a log to `logs/install-<time>.log` under the install location.

| Step | Action |
|---:|---|
| 1 | Runs `doctor.sh --pre`. On `FAIL` it stops and prints the ROCm commands |
| 2 | Creates the virtual environment |
| 3 | Installs the build dependencies, with the versions from `constraints/rocm-7.2.txt` |
| 4 | Installs PyTorch for ROCm 7.2, torchvision, Triton, and `amdsmi` from the ROCm directory |
| 5 | Fetches upstream vLLM at commit `c296851a7d` and applies `patches/vllm-rdna2-tp-0.1.0.patch` |
| 6 | Installs the requirements and builds vLLM in place. The native build took about 10 minutes on 48 threads. A repeated run first removes everything untracked in that source tree, so the build starts from zero |
| 7 | Checks that vLLM and PyTorch import, the GPUs are visible, `gfx1030` is in the architecture list, the host all-reduce module imports, and `vllm --help` runs. It then writes the marker `.installed-0.1.0` |

The installation has to stay editable, and `hipcc` has to stay installed: the `.hip` kernels are compiled from the source tree while the server runs. Do not build a wheel from this tree.

**Options**

| Option | Effect |
|---|---|
| `--prefix DIR` | Install location. Default: `$VLLM_RDNA2_HOME`, or `~/.local/share/vllm-rdna2`. The other scripts need `VLLM_RDNA2_HOME` set to the same directory |
| `--python BIN` | Python for the virtual environment. Default: `python3.12` |
| `--jobs N` | Parallel build jobs. Default: the number of CPUs |
| `--dry-run` | Prints every command and changes nothing. Only the read-only system check runs |
| `--skip-checks` | Turns a failed system check into a warning |
| `--force` | Removes the virtual environment and the source tree first, then reinstalls. `models/` and `logs/` are kept |

**Exit codes**

| Code | Meaning | Next step |
|---:|---|---|
| 0 | Installed, or already installed | |
| 2 | Bad call | See `./install.sh --help` |
| 3 | System requirements not met | Read the `FAIL` lines, or the message about Python |
| 4 | A dependency install failed | Check the network, then see the log |
| 5 | The source or the patch could not be obtained or applied | Check access to GitHub. A patch failure means a damaged checkout: clone again |
| 6 | The build failed | The last lines of the log give the cause. Check `hipcc`: `$ROCM_PATH/bin/hipcc --version` |
| 7 | The check after installation failed | Run `scripts/doctor.sh`. If no GPU is visible, check the `render` and `video` groups and log in again |

A second run finds the marker and changes nothing.

**Files after installation**

```
~/.local/share/vllm-rdna2/
  .installed-0.1.0    marker
  venv/               Python environment
  src/vllm/           patched vLLM, installed in place
  models/             weights downloaded by scripts/download-model.sh
  logs/               installer and server logs
  run/vllm.pid        PID of the running server
```

**Other install locations and ROCm paths.** Export them for all the scripts:

```bash
export VLLM_RDNA2_HOME=/data/vllm-rdna2
export ROCM_PATH=/opt/rocm    # the default
```

## 4. Download a model

```bash
scripts/download-model.sh --help
scripts/download-model.sh qwen3.6-35b-a3b --dry-run    # show the commands
scripts/download-model.sh qwen3.6-35b-a3b
```

| Preset | Repository | Download | Extra steps |
|---|---|---:|---|
| `qwen3.8-27b` | `philbert440/Qwen3.8-27B-W4A16-AWQ` | 19.6 GB | |
| `qwen3.5-9b` | `atalsharma/Qwen3.5-9B-AWQ` | 7.7 GB | For MTP, add the head as below |
| `qwen3.5-35b-a3b` | `cyankiwi/Qwen3.5-35B-A3B-AWQ-4bit` | 24.5 GB | |
| `qwen3.6-35b-a3b` | `cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit` | 25.0 GB | Converted automatically on the CPU, about 24 GB more |
| `gemma-4-26b-a4b` | `cyankiwi/gemma-4-26B-A4B-it-qat-AWQ-INT4` | 17.2 GB | |

Sizes are those of the Hugging Face repositories on 4 November 2026. The weights are not part of this project: each model has its own license on its page.

`download-model.sh` writes into `models/PRESET`, and `serve.sh` finds the weights there. A converted preset ends in `models/PRESET-ct-int4`. For another directory use `--dir DIR`, then pass the final directory to `serve.sh --model`. Exit codes: 0 ready; 1 the download, conversion or check failed; 2 bad call; 3 not installed.

**The MTP head for Qwen3.5-9B.** The AWQ repository has no MTP head. This command fetches the 15 `mtp.*` tensors from `Qwen/Qwen3.5-9B` with range requests, writes `model-mtp.safetensors` and updates `config.json`, keeping a copy as `config.json.before-mtp`:

```bash
scripts/add-mtp-head.py ~/.local/share/vllm-rdna2/models/qwen3.5-9b
```

## 5. Start the server

```bash
./serve.sh --list                 # presets, and whether their weights are present
./serve.sh qwen3.6-35b-a3b        # the first start took 5-11 minutes on the reference machine, later starts less
```

`serve.sh` runs in this order:

1. Checks the driver with `scripts/gpu-health.sh`.
2. Stops a previous server, and removes orphaned shared-memory files.
3. Starts the server in the background.
4. Waits for `/v1/models`, and sends a first short chat request as a probe.
5. On failure it stops what the attempt left, checks the driver and tries again, 3 times by default. A rank that does not stop ends the retries with code 6, unless you pass `--kill-stuck`.

The output gives the address, the model, the size of the KV cache and the log file. All options are in [CONFIGURATION.md](CONFIGURATION.md).

A preset without local weights downloads them from Hugging Face at the first start, with no progress on the screen. Use `download-model.sh` first: one attempt may take 1,200 s (`--start-timeout`), and a longer download fails the attempt.

## 6. Check it

```bash
python3 scripts/smoke-test.py
```

This sends one streamed chat request and checks that the answer is not empty and that usage is reported. It prints the answer, the time to the first token and the decode speed. Exit codes: 0 passed; 1 a check failed; 2 bad call; 3 the server could not be reached.

To check retrieval from a long context, use a prompt of about 100,000 tokens with a number hidden at its start. This takes minutes:

```bash
python3 scripts/smoke-test.py --needle 100000 --timeout 1800
```

## 7. Stop it

```bash
./stop.sh
```

`stop.sh` sends SIGTERM and waits up to 60 s (`--timeout S`). It returns 1 if the process survives, and it never sends SIGKILL. **Do not use `kill -9`**: it can hang the driver and freeze the machine. See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#the-process-does-not-exit).

## Running as a service

`install-service.sh` prints a systemd unit and installs nothing:

```bash
scripts/install-service.sh qwen3.6-35b-a3b --streams 4 > vllm-rdna2-qwen3.6-35b-a3b.service
head -12 vllm-rdna2-qwen3.6-35b-a3b.service    # the installation commands, as comments
```

Read the unit, then run the commands from its header. The unit:

- runs `serve.sh PRESET --foreground`, with no probe and no retries inside the script;
- restarts on failure after 60 s, at most twice per hour;
- does not restart on exit codes 2, 3, 4 and 6, because a person has to look at those first;
- stops with `stop.sh`, and `SendSIGKILL=no` keeps systemd from using SIGKILL.

## Update and removal

Reinstall or move to a new version:

```bash
./stop.sh
git pull
./install.sh --force
```

Remove everything, including the downloaded weights:

```bash
./stop.sh && rm -rf ~/.local/share/vllm-rdna2
```

If you installed a unit, first run `sudo systemctl disable --now vllm-rdna2-PRESET.service` and remove it from `/etc/systemd/system/`. ROCm stays installed; AMD's `amdgpu-uninstall` removes it.

## Tests

The scripts have tests that need neither GPUs nor network:

```bash
timeout 900 python3 -B -I -m unittest discover -s tests -v
```

See [tests/README.md](../tests/README.md).
