#!/usr/bin/env bash
# Mitigations for silent GPU hangs. REQUIRES A REBOOT.
#
# Symptom (2026-08-18, 16:12): the framebuffer froze and the log ended without a single entry,
# no oops and no page fault. The display is attached to 04:00.0 (DRM card1), the same card
# that computes. When it hangs, both the desktop and the logging die, so no evidence remains.
#
# What it adds:
#   amdgpu.gpu_recovery=1  - force a reset of a hung card. The default (-1, "auto") often does
#                            not trigger on consumer RDNA2 when the card also draws the screen.
#   amdgpu.lockup_timeout=10000,10000,10000,10000
#                          - final timeout (ms) for gfx/compute/sdma/video, so the driver notices
#                            the stall and tries recovery instead of waiting forever.
#   nmi_watchdog=1
#   kernel.panic_on_hardlockup - turn a dead hang into a panic: the system reboots by itself and
#                            leaves a record in pstore instead of standing frozen.
#
# Run as root: sudo ./gpu-hang-mitigations.sh
set -eu

if [ "$(id -u)" -ne 0 ]; then
    echo "gpu-hang-mitigations: must run as root (for example: sudo $0)" >&2
    exit 2
fi

PARAMS="amdgpu.gpu_recovery=1 amdgpu.lockup_timeout=10000,10000,10000,10000 nmi_watchdog=1"

BAK=/etc/default/grub.bak.$(date +%Y%m%d-%H%M%S)
cp /etc/default/grub "$BAK"
echo "backup: $BAK"

for p in $PARAMS; do
    key=${p%%=*}
    if grep -q "$key" /etc/default/grub; then
        echo "  $key already present, skipping"
    else
        sed -i "s|^\(GRUB_CMDLINE_LINUX_DEFAULT=\"[^\"]*\)\"|\1 $p\"|" /etc/default/grub
        echo "  added $p"
    fi
done
grep GRUB_CMDLINE_LINUX_DEFAULT /etc/default/grub
timeout 300 update-grub

# Turn a dead hang into a panic: the system reboots by itself and leaves a trace
# instead of standing frozen.
#   hardlockup_panic    - the kernel stopped answering (needs nmi_watchdog=1, so the key
#                         appears only after a reboot)
#   hung_task_panic     - a task stayed in uninterruptible sleep (D state) longer than
#                         hung_task_timeout; this is what the hang at 05:41 looked like
#                         (kfd_process_wq_release)
tee /etc/sysctl.d/99-gpu-hang.conf >/dev/null <<'SYSCTL'
kernel.hardlockup_panic = 1
kernel.hung_task_panic = 1
kernel.hung_task_timeout_secs = 120
kernel.panic = 20
SYSCTL
# -e: until the reboot, panic_on_hardlockup does not exist yet; that is not an error
timeout 30 sysctl -e -p /etc/sysctl.d/99-gpu-hang.conf

echo
echo "Done. Takes effect after a reboot."
echo "After the next hang check: journalctl --list-boots; ls /sys/fs/pstore/"
