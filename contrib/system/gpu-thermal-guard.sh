#!/usr/bin/env bash
# GPU overheat guard for two Navi 21 cards.
#
# Why. On 2026-08-18 at 17:23 the system shut itself down:
#   amdgpu 0000:04:00.0: ERROR: GPU over temperature range(SW CTF) detected!
#   amdgpu 0000:04:00.0: ERROR: System is going to shutdown due to GPU SW CTF!
# The stock fan curve follows the edge temperature, not the junction. Measured:
# at junction 95 C the edge read only 76 C, so the fan stayed at 40-45% and the
# junction temperature crept up to the CTF limit (about 110 C).
#
# The edge-to-junction gap here is 17-19 C and does not disappear even at full fan
# speed, so the fan alone is not enough. The guard also limits the clock (see
# gpu-tune.sh, SCLK_MAX) and drops the clock in an emergency.
set -u

POLL=${POLL:-2}
SCLK_NORMAL=${SCLK_NORMAL:-1600}
SCLK_PANIC=${SCLK_PANIC:-1000}
T_PANIC=${T_PANIC:-93}      # junction temperature, C: emergency clock drop
T_CLEAR=${T_CLEAR:-85}      # C: back to the normal clock

# ── Fixed fan speed on request ────────────────────────────────────────────
# If the file FIKS_FAJL (the name is kept from the original) contains a number 0..255,
# it is used instead of the curve, WHILE the junction temperature is below T_FIKS_MAX.
# Above it the curve applies again: otherwise the only protection against SW CTF, which
# already shut the machine down once, would be removed. The file lives in /run, so a
# reboot clears the mode by itself.
FIKS_FAJL=${FIKS_FAJL:-/run/gpu-fan-fixed}
T_FIKS_MAX=${T_FIKS_MAX:-89}

# sysfs roots; overridable only for tests on a fake tree.
DRM=${DRM:-/sys/class/drm}
HWMON=${HWMON:-/sys/class/hwmon}

fixed_speed() {
    local v
    [ -r "$FIKS_FAJL" ] || return 1
    v=$(tr -dc '0-9' < "$FIKS_FAJL")
    [ -n "$v" ] && [ "$v" -ge 0 ] && [ "$v" -le 255 ] || return 1
    echo "$v"
}

# junction -> pwm (0..255). Much more aggressive than the stock curve.
pwm_for() {
    local t=$1
    if   [ "$t" -lt 60 ]; then echo -1        # -1 = hand control back to the driver
    elif [ "$t" -lt 70 ]; then echo 100
    elif [ "$t" -lt 78 ]; then echo 150
    elif [ "$t" -lt 84 ]; then echo 190
    elif [ "$t" -lt 89 ]; then echo 225
    else                       echo 255
    fi
}

# Current values from the card's own table.
od_sclk() { awk '/OD_SCLK:/ {getline; getline; gsub(/[^0-9]/, "", $2); print $2; exit}' "$1" 2>/dev/null; }
od_vo()   { awk '/OD_VDDGFX_OFFSET:/ {getline; gsub(/[^0-9-]/, "", $1); print $1; exit}' "$1" 2>/dev/null; }

# ⚠️ The voltage offset is NOT written as a number: the guard writes back whatever the card
# already holds (gpu-tune.sh sets it from VOLT_OFFSET). Until 2026-09-23 the value `vo -30`
# was hard-coded here, so if VOLT_OFFSET were set back to 0, the first panic or recovery
# would silently restore -30.
#
# ⚠️ Every write is verified by reading it back, with retries. gpu-tune.sh already knows
# that writes to pp_od_clk_voltage can silently have no effect. The guard did not check
# this and treated the clock as dropped even when the write had failed, and that is its
# only protection against SW CTF apart from the fan.
set_sclk() {
    local mhz=$1 od D vo attempt
    for od in "$DRM"/card*/device/pp_od_clk_voltage; do
        [ -e "$od" ] || continue
        D=$(dirname "$od")
        vo=$(od_vo "$od")
        for attempt in 1 2 3; do
            echo manual     > "$D/power_dpm_force_performance_level" 2>/dev/null
            echo "s 1 $mhz" > "$od" 2>/dev/null
            [ -n "$vo" ] && echo "vo $vo" > "$od" 2>/dev/null
            echo c          > "$od" 2>/dev/null
            echo auto       > "$D/power_dpm_force_performance_level" 2>/dev/null
            [ "$(od_sclk "$od")" = "$mhz" ] && break
            sleep 0.5
        done
        if [ "$(od_sclk "$od")" != "$mhz" ]; then
            logger -t gpu-thermal-guard "FAILED to set SCLK ${mhz} on $D (now $(od_sclk "$od"))"
        fi
    done
}

panicked=0
declare -A unreadable=()     # log only changes of state, not every poll
while :; do
    hottest=0
    for h in "$HWMON"/hwmon*; do
        [ "$(cat "$h/name" 2>/dev/null)" = "amdgpu" ] || continue
        # ⚠️ An unreadable sensor is NOT 0 C. Until 2026-09-23 a failed read gave t=0,
        # which meant "hand control to the driver" (control by the edge temperature, the
        # very thing that led to SW CTF on 2026-08-18), and the panic never fired. Now such
        # a card is treated as hot: full fan speed and a clock drop.
        raw=$(cat "$h/temp2_input" 2>/dev/null)
        case "$raw" in
            ''|*[!0-9]*)
                t=$T_PANIC
                [ -n "${unreadable[$h]:-}" ] || logger -t gpu-thermal-guard \
                    "cannot read $h/temp2_input, assuming ${T_PANIC}C, fan to maximum"
                unreadable[$h]=1 ;;
            *)  t=$(( raw / 1000 ))
                [ -n "${unreadable[$h]:-}" ] && logger -t gpu-thermal-guard "$h/temp2_input is readable again"
                unset "unreadable[$h]" ;;
        esac
        [ "$t" -gt "$hottest" ] && hottest=$t
        p=$(pwm_for "$t")
        if f=$(fixed_speed) && [ "$t" -lt "$T_FIKS_MAX" ]; then
            p=$f
        fi
        if [ "$p" -lt 0 ]; then
            echo 2 > "$h/pwm1_enable" 2>/dev/null
        else
            echo 1 > "$h/pwm1_enable" 2>/dev/null
            echo "$p" > "$h/pwm1" 2>/dev/null
        fi
    done

    if [ "$panicked" = 0 ] && [ "$hottest" -ge "$T_PANIC" ]; then
        logger -t gpu-thermal-guard "junction ${hottest}C >= ${T_PANIC}C, dropping SCLK to ${SCLK_PANIC}"
        set_sclk "$SCLK_PANIC"; panicked=1
    elif [ "$panicked" = 1 ] && [ "$hottest" -le "$T_CLEAR" ]; then
        logger -t gpu-thermal-guard "junction ${hottest}C <= ${T_CLEAR}C, restoring SCLK ${SCLK_NORMAL}"
        set_sclk "$SCLK_NORMAL"; panicked=0
    fi
    sleep "$POLL"
done
