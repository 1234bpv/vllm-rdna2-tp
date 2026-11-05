#!/usr/bin/env bash
# Undervolt and clock limit for both Navi 21 cards.
#
# ⚠️ MCLK is NOT changed. On these cards any value above the stock 1000 MHz switches the
# memory timings and cuts generation speed by 26% (18.98 -> 14.03 t/s).
#
# ⚠️ Writes to pp_od_clk_voltage silently have no effect when made too soon after boot: the
# driver accepts them without an error, but the values stay at the stock ones (2026-08-18: the
# service reported "Finished", the offset was 0 mV and SCLK 2369 MHz). So each write is checked
# by reading it back, with retries.
set -u

SCLK_MAX=${SCLK_MAX:-1600}      # MHz. 1900 gave 95 C junction and an emergency shutdown (SW CTF); 1600 holds 85-86 C at full fan speed.
# ⚠️ The undervolt was RESTORED on 2026-08-25 after an experiment from 2026-08-21 to 2026-08-25.
# The offset had been left at 0 from 2026-08-21 because two hangs (21:25 and 21:42 on 2026-08-21)
# happened at IDLE, with no vLLM and no model loaded. The undervolt fails first in low DPM states,
# where the voltage margin is smallest, which is exactly idle. The suspicion was NOT confirmed: the
# real cause of the instability was the CPU's C6 state, not the card offset. So -30 mV is restored
# on both cards (it gave a noticeably lower temperature). If idle hangs return, the C6 suspicion is
# wrong: set the offset back to 0 and keep looking.
VOLT_OFFSET=${VOLT_OFFSET:--30}  # mV

apply_one() {
    local od=$1 D
    D=$(dirname "$od")
    echo manual          > "$D/power_dpm_force_performance_level" 2>/dev/null
    echo "s 1 $SCLK_MAX" > "$od" 2>/dev/null
    echo "vo $VOLT_OFFSET" > "$od" 2>/dev/null
    echo "c"             > "$od" 2>/dev/null
    echo auto            > "$D/power_dpm_force_performance_level" 2>/dev/null
    # The offset is compared as a number, not as a string: the driver may print a zero offset
    # without "0mV", and a literal grep would turn an unexpected format into 30 useless retries
    # and a FAILED message for a write that actually succeeded.
    #
    # ⚠️ Do not use a placeholder such as ${got:-999} here: an empty `got` would become "999 != 0",
    # a false failure exactly when VOLT_OFFSET=0. And with VOLT_OFFSET=999 the same placeholder
    # would give a false SUCCESS. So an empty value is handled separately, by its meaning:
    #   zero offset     - nothing to apply, the stock value is 0: success;
    #   nonzero offset  - a real change was required and cannot be confirmed: failure, and it
    #                     must not be silent.
    local got
    got=$(awk '/OD_VDDGFX_OFFSET/{getline; print}' "$od" | grep -oE '^-?[0-9]+')
    grep -q "1: ${SCLK_MAX}Mhz" "$od" || return 1
    if [ -z "$got" ]; then
        [ "$VOLT_OFFSET" -eq 0 ] || return 1
        echo "$D: offset is not readable from sysfs, but zero is required; taking it as the stock value" >&2
        return 0
    fi
    [ "$got" -eq "$VOLT_OFFSET" ]
}

rc=0
for od in /sys/class/drm/card*/device/pp_od_clk_voltage; do
    ok=0
    for attempt in $(seq 30); do
        if apply_one "$od"; then ok=1; break; fi
        sleep 2
    done
    if [ "$ok" = 1 ]; then
        echo "$(dirname "$od"): SCLK<=${SCLK_MAX} MHz, offset ${VOLT_OFFSET} mV - applied"
    else
        echo "$(dirname "$od"): FAILED to apply within 60 s" >&2
        rc=1
    fi
done
exit $rc
