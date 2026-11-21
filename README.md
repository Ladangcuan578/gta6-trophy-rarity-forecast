# GTA VI Trophy Rarity Forecast

[![make.fun](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ladangcuan578/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/Ladangcuan578/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

Entry for the make.fun bounty [Forecast how many GTA VI players earn each
trophy in week one](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
a program (`main.py`, Python 3.13, standard library only) that reads one case
from `/input` and writes, for each of the game's 55 PlayStation 5 trophies,
five quantiles of its earn rate on Exophase at the case's target, scored by
pinball loss on the logit scale.

## Model: holder counts, not rates

Exophase prints each trophy's earn rate and the number of players tracked.
Write `rate = H / N`, with `H` the holders of a trophy and `N` the players
tracked. Both are counts and only ever grow; the rate moves because the two
grow at different speeds.

The page does not move continuously. Exophase refreshes a game's counts in
jumps, hours apart: between refreshes every read is identical, and at a
refresh `N` jumps to the players who have synced since while each `H` jumps
to the holders among them plus the trophies the old players earned since. A
refresh therefore usually *lowers* most rates (new players hold fewer
trophies) while the trophies every newcomer already holds stay put. Launch
night showed this once: `N` 638 → 3,868 in one refresh, trophies earned
943 → 4,276, every rate but the one at 100% down.

So the forecast is built from counts:

1. Find the refreshes in the history (reads where `N` or any rate changed).
2. From them, learn how fast `N` grows (its growth rate decays
   exponentially as the launch wave passes), how often a refresh lands, and,
   per trophy, how many of a refresh's new players already hold it (the
   cohort hold rate) and how fast the old players keep earning it.
3. Carry `N` and every `H` forward to the target and read the rates off as
   `H / N`. Story trophies keep being earned all week; grind and collection
   trophies barely move; the trophy's kind (from its tier and description)
   sets the prior for both.
4. Widen the spread with the horizon, the number of refreshes ahead and the
   distance the counts are carried, because a 5-day carry is far less
   certain than a 90-minute one.

When the history is flat (no refresh yet, as on launch night) the counts
cannot be learned, so the program falls back on priors from the launch-night
refresh the kit documents: a refresh about every 4.5 hours, `N` roughly six
times a refresh while the base is small, and a new cohort that holds the
trophy everybody holds and only a sliver of the rest.

## Practice results

The kit does not ship the practice answers (the 04:30Z and 05:00Z reads), so
the practice loss below is against a reconstructed answer: the refresh's
totals are the kit's own figures (`N` 3,868, 4,276 trophies earned), the
100% trophy is held by every new player, and the remaining 103 new trophies
are spread over the other trophies. That reconstruction reproduces the
house baseline's published practice figures almost exactly (persistence loss
1.39–1.42 on it, typical miss 0.68 points, coverage 0 — the spec says 1.51,
0.68, 0), so it is a close stand-in for the real answer.

| forecast | practice loss | typical miss | 90% coverage |
|---|---:|---:|---:|
| persistence baseline (house) | 1.51 | 0.68 pp | 0.00 |
| this program | **≈ 0.10** | 0.02 pp | 0.98 |

The program also beats the baseline under a noisier simulation of the
answer (multinomial draws with over-dispersion): loss ≈ 0.14 vs ≈ 1.43 for
persistence, in every scenario tried. Runtime is ~0.1 s per case (cap 8 s),
output is deterministic (two runs byte-identical), and the format validates
(55 rows, quantiles non-decreasing, 0–100, ≤ 65,536 bytes).

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

## License

MIT — see `LICENSE`.
