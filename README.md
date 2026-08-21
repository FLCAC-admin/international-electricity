# international-electricity

Generating international electricity mixes for the [Federal LCA Commons](lcacommons.gov).

Country mixes link Ember fuels to **US-average** resource processes. Build those first with [us_fuel_generation_mixes.py](us_fuel_generation_mixes.py) (BA-weighted averages from the US Electricity Baseline).

Zips land in `output/`. Extracts: `output/us_fuel_generation_mixes_v1.0/` and `output/international_electricity_v1.0.0/`.

Each country (except USA) then gets two processes in the same international zip:

- **At grid:** generation mix from [Ember yearly electricity data](https://ember-energy.org/data/yearly-electricity-data/).
- **At user; generation mix:** that mix scaled up by transmission and distribution losses from World Bank [EG.ELC.LOSS.ZS](https://data.worldbank.org/indicator/EG.ELC.LOSS.ZS) (trade is not modeled). Latest loss year ≤ mix year if used. If no country series exists, the World Bank geographic region is used. Reference flow is U.S. Electricity, AC, 120 V (low voltage proxy);
high voltage input is the country at-grid process at `1 / (1 − loss)`.

See the [list of countries](country_list.md).

**Import order:** (1) US Electricity Baseline, (2) US-average fuel zip, (3) international at-grid + at-user zip.

## Install

Requires [flcac-utils](https://github.com/FLCAC-admin/flcac-utils).

> pip install git+https://github.com/FLCAC-admin/flcac-utils.git

Run [us_fuel_generation_mixes.py](us_fuel_generation_mixes.py). Losses are committed as `data/eg_elc_loss_zs.csv`. Refresh with [download_worldbank_td_losses.py](download_worldbank_td_losses.py), then run [international_electricity_mixes_olca.py](international_electricity_mixes_olca.py).

## Datasets

| Datasts                                      | Version | flcac-utils | Release        |
|----------------------------------------------|---------|-------------|----------------|

<!--| International Electricity mixes (2022, 2023) | v1.0.0  | v0.1.0      | 2025 Q1, USLCI | -->
