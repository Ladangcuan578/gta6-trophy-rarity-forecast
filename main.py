"""GTA VI trophy rarity forecast (gta6-trophy-rarity-forecast-01).

A holder-count forecaster.  Exophase prints, for each trophy, the share of
the players it tracks who hold it, and the number of players tracked.  Write
    rate_i(t) = H_i(t) / N(t)
with H_i the number of tracked players holding trophy i and N the number of
tracked players.  Both are counts, so both only ever grow, and the rate moves
only because the two grow at different speeds.

The page does not move continuously.  Exophase refreshes a game's counts in
jumps, hours apart: between refreshes every read is byte-identical, and at a
refresh N jumps to the players who have synced since, and each H_i jumps to
the holders among them plus the trophies the old players earned since.  A
refresh therefore usually *lowers* most rates (the new players hold fewer
trophies than the old ones) while the few trophies every newcomer already
holds stay put.  Launch night showed this once: N 638 -> 3,868 in one
refresh, trophies earned 943 -> 4,276, every rate but the one at 100% down.

So the forecast is built from counts, not from extrapolating the rates:
  1. find the refreshes in the history (reads where N or any rate changed);
  2. from them, learn how fast N grows, how often a refresh lands, and, per
     trophy, how many of a refresh's new players already hold it (the cohort
     hold rate) and how fast the old players keep earning it;
  3. carry N and every H_i forward to the target with those rates, letting
     the growth of N slow as the launch wave passes, and read the rates off
     as H_i / N;
  4. widen the spread with the horizon, the number of refreshes ahead and
     the distance the counts have to be carried, because a 5-day carry is
     far less certain than a 90-minute one.

When the history is flat (no refresh yet, as on launch night) the counts
cannot be learned, so the program falls back on priors taken from the
launch-night refresh the kit documents: a refresh about every 4.5 hours,
N roughly six times a refresh while the base is small, and a new cohort
that holds the trophy everybody holds and only a sliver of the rest.

Standard library only, no network, no randomness: two runs on the same
case write the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# --- priors, from the launch-night refresh the kit documents -----------------
# A refresh lands about every 4.5 h; while the tracked base is small one
# refresh multiplies it by about six (638 -> 3,868).  A new cohort holds the
# trophy everybody holds, and of any other trophy only a sliver, roughly
# proportional to how common it already is.
PRIOR_INTERVAL_H = 4.5
PRIOR_N_MULT = 6.0
PRIOR_COHORT_SLOPE = 0.07
# Random-walk spread of a rate on the logit scale, per square-root day.
WALK_SIGMA = 0.5
# Extra logit spread per refresh ahead (the jump itself is uncertain) and
# for carrying the model far past the history.
REFRESH_SPREAD = 0.45
MODEL_SPREAD = 0.30
# A new cohort's hold rate fades over a long carry: later joiners are more
# casual than the launch-night ones.
COHORT_DECAY_DAYS = 6.0


def parse_time(text):
    """An ISO 8601 UTC timestamp to a timezone-aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def clamp_rate(pct):
    return min(99.99, max(0.01, pct))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def inv_logit(x):
    return 100.0 / (1.0 + math.exp(-x))


def round4(value):
    """Round to 4 decimals, halves away from zero, as a plain number."""
    q = Decimal(str(value)).quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    out = float(q)
    return int(out) if out == int(out) else out


def norm_ppf(p):
    """The standard normal quantile at p (Acklam's rational approximation).

    Accurate to about 1e-9, which is far finer than any rate here needs.
    """
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


# --- what kind of trophy this is, from its tier and description -------------
# Story trophies keep being earned all week (players finish chapters); grind
# and collection trophies barely move (few players ever finish them); the
# rest sit between.  The kind sets how fast old players earn it, how much
# of a new cohort already holds it, and how much of the non-holders can
# plausibly earn it over the horizon.

_STORY_WORDS = ("chapter", "final mission", "complete gone fishing",
                "complete breakfast run", "complete the cartel heist",
                "complete child's play", "complete headliners",
                "complete vcia takedown", "complete operation savage glory")
_GRIND_WORDS = ("100% completion", "250 stunts", "gold medal", "drone race",
                "20 different races", "100 vehicles", "50 challenges",
                "collectible card", "100 locations", "every spot",
                "explore all of leonida", "all garage properties",
                "10 vehicles to the", "laundering", "across all shops",
                "bank vaults", "armored truck", "illegal operation",
                "8 mission strands", "every location")


def trophy_kind(trophy):
    """'story', 'grind' or 'misc', from the trophy's tier and description."""
    desc = (trophy.get("description") or "").lower()
    if trophy.get("tier") == "platinum":
        return "grind"
    if any(w in desc for w in _STORY_WORDS):
        return "story"
    if any(w in desc for w in _GRIND_WORDS):
        return "grind"
    return "misc"


# Per kind: the daily hazard for a non-holder to earn it, and the largest
# share of the non-holders that can plausibly earn it over the horizon.
# The hazards are forward rates, not averages since launch: the players
# who have not earned a trophy yet are the slower ones (the keen ones
# already hold it), so the forward hazard is well below the average that
# produced the current rate.
KIND_PRIOR = {
    "story": {"earn_per_day": 0.03, "cap_frac": 0.45},
    "misc": {"earn_per_day": 0.015, "cap_frac": 0.15},
    "grind": {"earn_per_day": 0.003, "cap_frac": 0.04},
}


# --- reading the history ----------------------------------------------------

def refresh_points(case):
    """One (time, players, holders) per refresh: the reads where the page moved.

    Between refreshes every read is identical, so only the first read of
    each distinct page counts.  Holders are recovered from the printed
    rate: holders = rate / 100 * players, rounded to the nearest whole
    player (the page prints 2 decimals, so this is nearly exact).
    """
    points = []
    prev = None
    for read in case["history"]:
        if prev is not None and read["players_tracked"] == prev["players_tracked"] \
                and read["rates"] == prev["rates"]:
            continue
        holders = []
        for rate in read["rates"]:
            if rate is None:
                holders.append(None)
            else:
                holders.append(int(round(rate / 100.0 * read["players_tracked"])))
        points.append((parse_time(read["at"]), read["players_tracked"], holders))
        prev = read
    return points


def fit_log_growth(points, target_time):
    """Carry the tracked-player count to the target from the refreshes.

    A launch's tracked base grows fast and then slows: the growth rate of
    log N itself decays, roughly exponentially, as the launch wave passes.
    So the carry projects the current growth rate forward with that decay:
    the rate r now, decaying with time constant tau, adds
    r * tau * (1 - exp(-horizon / tau)) to log N.  The current rate comes
    from the most recent day of refreshes; tau comes from how much the
    rate has already fallen over the history (a prior of 2 days when the
    history cannot show it).  Returns (N_target, n_refreshes_ahead) with
    N_target at least the last N.
    """
    times = []
    logs = []
    t0 = points[0][0]
    for t, n, _ in points:
        times.append((t - t0).total_seconds() / 86400.0)
        logs.append(math.log(max(1, n)))
    horizon_days = (target_time - points[-1][0]).total_seconds() / 86400.0
    n_last = points[-1][1]
    if len(points) < 2 or horizon_days <= 0:
        return float(n_last), 0
    t_last = times[-1]

    def rate_over(days_back):
        """Log-N growth rate over the window ending at the last point."""
        pts = [(t, l) for t, l in zip(times, logs) if t >= t_last - days_back]
        if len(pts) < 2:
            return None
        t_a, l_a = pts[0]
        t_b, l_b = pts[-1]
        if t_b - t_a < 1e-9:
            return None
        return (l_b - l_a) / (t_b - t_a)

    r_now = rate_over(1.0)
    if r_now is None:
        r_now = rate_over(t_last)
    if r_now is None or r_now <= 0:
        return float(n_last), 0
    # How fast the rate itself has been falling: compare the first day's
    # rate with the last day's.  tau is the time for the rate to fall by e.
    r_early = rate_over(t_last) if t_last > 2.0 else None
    tau = 2.0
    if r_early is not None and r_early > r_now > 0:
        span = max(0.5, t_last - 1.0)
        tau = span / math.log(r_early / r_now)
        tau = min(max(tau, 0.5), 10.0)
    log_add = r_now * tau * (1.0 - math.exp(-horizon_days / tau))
    n_tgt = math.exp(logs[-1] + log_add)
    n_tgt = max(float(n_last), min(n_tgt, float(n_last) * 50.0))
    gaps = [(points[k + 1][0] - points[k][0]).total_seconds() / 3600.0
            for k in range(len(points) - 1)]
    gap = sorted(gaps)[len(gaps) // 2] if gaps else PRIOR_INTERVAL_H
    gap = min(max(gap, 0.5), 24.0)
    n_ref = max(0, int(horizon_days * 24.0 / gap))
    return n_tgt, n_ref


def cohort_and_earn(case, points, kinds):
    """Per trophy, the cohort hold rate and the old players' earning rate.

    At each refresh, dH / dN is the share of the refresh's new players who
    already hold the trophy.  What is left of dH, after the new cohort's
    share at the prior rate is removed, spread over the non-holders and
    the time since the last refresh, is the old players earning it.  Both
    are averaged over refreshes with recent ones weighing more, then shrunk
    toward the kind's prior (the shrinkage is what a flat history leaves:
    the prior alone).
    Returns (cohort[i], earn[i]) for each trophy.
    """
    n_troph = len(case["trophies"])
    last_rates = case["history"][-1]["rates"]
    cohort = [None] * n_troph
    earn = [None] * n_troph
    if len(points) < 2:
        return cohort, earn
    for i in range(n_troph):
        kind = kinds[i]
        prior = KIND_PRIOR[kind]
        last_rate = last_rates[i] if last_rates[i] is not None else 0.0
        prior_cohort = PRIOR_COHORT_SLOPE * (last_rate / 100.0)
        prior_earn = prior["earn_per_day"]
        num_c = 0.0
        den_c = 0.0
        num_e = 0.0
        den_e = 0.0
        for k in range(1, len(points)):
            t_prev, n_prev, h_prev = points[k - 1]
            t_cur, n_cur, h_cur = points[k]
            if h_prev[i] is None or h_cur[i] is None:
                continue
            dn = n_cur - n_prev
            dh = h_cur[i] - h_prev[i]
            dt_days = (t_cur - t_prev).total_seconds() / 86400.0
            if dn <= 0 or dt_days <= 0:
                continue
            age_days = (points[-1][0] - t_cur).total_seconds() / 86400.0
            w = 0.5 ** (age_days / 1.0)
            num_c += w * (dh / dn)
            den_c += w
            nonholders = max(1, n_prev - h_prev[i])
            residual = max(0.0, dh - prior_cohort * dn)
            num_e += w * (residual / (nonholders * dt_days))
            den_e += w
        if den_c > 0:
            shrink = den_c / (den_c + 1.0)
            cohort[i] = shrink * (num_c / den_c) + (1.0 - shrink) * prior_cohort
        else:
            cohort[i] = prior_cohort
        if den_e > 0:
            shrink = den_e / (den_e + 1.0)
            earn[i] = shrink * (num_e / den_e) + (1.0 - shrink) * prior_earn
        else:
            earn[i] = prior_earn
    return cohort, earn


def forecast_case(case):
    levels = case["levels"]
    zvals = [norm_ppf(tau) for tau in levels]
    trophies = case["trophies"]
    kinds = [trophy_kind(t) for t in trophies]
    history = case["history"]
    if not history:
        raise ValueError("case has no history")
    last = history[-1]
    n_last = last["players_tracked"]
    t_now = parse_time(case["at"])
    t_target = parse_time(case["target"])
    horizon_days = max(0.0, (t_target - t_now).total_seconds() / 86400.0)
    horizon_hours = horizon_days * 24.0
    points = refresh_points(case)

    # --- carry the tracked-player count to the target ----------------------
    if len(points) >= 2:
        n_target, n_ref = fit_log_growth(points, t_target)
        cohort, earn = cohort_and_earn(case, points, kinds)
    else:
        # Flat history: no refresh has been seen.  A refresh is due -- the
        # page has been flat since the first read -- so count the refreshes
        # the horizon spans at the prior interval, starting from the time
        # the next one is due (interval less the flat time already seen).
        t_first = parse_time(history[0]["at"])
        age_h = (t_now - t_first).total_seconds() / 3600.0
        t_next = max(0.25, PRIOR_INTERVAL_H - age_h)
        if horizon_hours <= t_next:
            n_ref = 0
        else:
            n_ref = 1 + int((horizon_hours - t_next) // PRIOR_INTERVAL_H)
        n_target = float(n_last) * (PRIOR_N_MULT ** n_ref)
        cohort = [None] * len(trophies)
        earn = [None] * len(trophies)

    # --- carry every trophy's holders to the target ------------------------
    rows = []
    for i, trophy in enumerate(trophies):
        rate = last["rates"][i]
        if rate is None:
            rate = 0.0
        holders = rate / 100.0 * n_last
        kind = kinds[i]
        prior = KIND_PRIOR[kind]
        if rate >= 99.99:
            # Everybody holds it, and every newcomer does too.
            med = 100.0
            spread = math.sqrt((WALK_SIGMA ** 2) * horizon_days + 0.01)
            lg = logit(med)
            qs = [inv_logit(lg + z * spread) for z in zvals]
        else:
            if cohort[i] is None:
                c_rate = PRIOR_COHORT_SLOPE * (rate / 100.0)
            else:
                c_rate = cohort[i]
            if earn[i] is None:
                e_rate = prior["earn_per_day"]
            else:
                e_rate = earn[i]
            # The new cohort's share fades over a long carry: later
            # joiners are more casual than the launch-night ones.
            fade = 1.0 / (1.0 + horizon_days / COHORT_DECAY_DAYS)
            c_rate = c_rate * fade
            # Old players keep earning it: holders grow toward N, capped
            # at the share of non-holders the kind can plausibly reach.
            nonholders = max(0.0, n_last - holders)
            growth = nonholders * (1.0 - math.exp(-e_rate * horizon_days))
            growth = min(growth, nonholders * prior["cap_frac"])
            h_target = holders + growth + (n_target - n_last) * c_rate
            h_target = max(0.0, min(h_target, n_target))
            med = h_target / n_target * 100.0
            # Spread: the random walk, the refresh jumps ahead, and the
            # model being carried far past what it has seen.
            spread = math.sqrt(
                (WALK_SIGMA ** 2) * horizon_days
                + (REFRESH_SPREAD ** 2) * math.log1p(n_ref)
                + (MODEL_SPREAD ** 2) * math.log1p(horizon_days)
            )
            lg = logit(med)
            qs = [inv_logit(lg + z * spread) for z in zvals]
        qs = [min(99.99, max(0.01, q)) for q in qs]
        for k in range(1, len(qs)):
            qs[k] = max(qs[k], qs[k - 1])
        rows.append({"id": trophy["id"], "q": [round4(q) for q in qs]})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
