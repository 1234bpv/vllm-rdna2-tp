# System files: examples for one test machine

These files are the system configuration of the test machine: two Navi 21 cards (gfx1030),
with the display attached to card 04:00.0. They are examples for that one machine.

- `install.sh` does not install them.
- The voltage, clock and fan values belong to that machine. Do not apply them blindly to
  another one.
- The measurements of this project were made with the clock limited to 1600 MHz and the
  voltage offset at -30 mV. These are the defaults of `gpu-tune.sh`.

## Files

| File | Where it goes | Purpose |
|---|---|---|
| `gpu-thermal-guard.sh` | `/usr/local/sbin/` | drives the fans by the junction temperature, not the edge; drops the clock in an emergency at 93 C |
| `gpu-thermal-guard.service` | `/etc/systemd/system/` | runs the guard |
| `gpu-tune.sh` | `/usr/local/sbin/` | clock limit 1600 MHz and voltage offset -30 mV, checked by reading back |
| `gpu-tune.service` | `/etc/systemd/system/` | applies the settings at boot |
| `99-gpu-hang.conf` | `/etc/sysctl.d/` | turns a dead hang into a panic with a reboot, so that a record remains |
| `gpu-hang-mitigations.sh` | run once, as root | adds kernel parameters to GRUB; needs a reboot |

## Installation

Run as root (the scripts do not call `sudo` themselves):

```bash
sudo cp gpu-tune.sh gpu-thermal-guard.sh /usr/local/sbin/
sudo chmod +x /usr/local/sbin/gpu-{tune,thermal-guard}.sh
sudo cp gpu-*.service /etc/systemd/system/
sudo cp 99-gpu-hang.conf /etc/sysctl.d/
sudo systemctl daemon-reload
sudo systemctl enable --now gpu-tune.service gpu-thermal-guard.service
sudo sysctl -e -p /etc/sysctl.d/99-gpu-hang.conf
```

## Hang mitigations

`sudo ./gpu-hang-mitigations.sh` adds the kernel parameters to GRUB and then overwrites
`/etc/sysctl.d/99-gpu-hang.conf` with a four-line version (it also sets `hung_task_panic`,
`hung_task_timeout_secs` and `panic`). Run it once, then reboot. If you use the script, skip
the `99-gpu-hang.conf` line of the installation above: the file in this directory is the
earlier two-line version.

After the next hang, check the previous boot:

```bash
journalctl --list-boots
ls /sys/fs/pstore/
```

## Status

These files were not run on the test machine as part of the release preparation. Only their
syntax was checked (`bash -n`).
