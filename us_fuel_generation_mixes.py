#!/usr/bin/env python3
"""
Aggregate balancing-authority electricity resource processes into US-average
resource processes (SOLAR, WIND, COAL, etc.) using the US grid consumption mix.

Method
1) Read BA weights from:
   Electricity; at grid; consumption mix - US - US
2) For each BA generation-mix provider in that US process, read the BA
   generation mix shares by resource process (e.g., Electricity - SOLAR - BA).
3) Compute resource contribution weights:
   contribution(BA, resource) = US_BA_weight * BA_resource_fraction
4) Normalize contribution weights within each resource so the selected BA
   resource providers sum to 1.0 for that resource.
5) Create one aggregated process per resource as a weighted mix of BA
   resource-process providers.

Inputs
- US Electricity Baseline processes from the Federal LCA Commons
  (flcac-utils ``read_commons_data``)

Outputs
- zip: ``output/us_fuel_generation_mixes_olca2.0_*.zip`` (same folder as international)
- extract + audit CSVs: ``output/us_fuel_generation_mixes/``
"""

from __future__ import annotations

import copy
import csv
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import olca_schema as olca
from esupy.util import make_uuid
from flcac_utils.commons_api import read_commons_data
from flcac_utils.generate_processes import write_objects
from flcac_utils.util import extract_latest_zip

BASE_DIR = Path(__file__).resolve().parent
ZIP_DIR = BASE_DIR / "output"
EXTRACT_DIR = BASE_DIR / "output" / "us_fuel_generation_mixes"

BASELINE_REPO_KEY = "US Electricity Baseline"
US_GRID_CONSUMPTION_MIX_NAME = "Electricity; at grid; consumption mix - US - US"
GRID_GENERATION_MIX_PREFIX = "Electricity; at grid; generation mix - "
RESOURCE_PROCESS_PATTERN = re.compile(r"^Electricity - ([^-]+?) - (.+)$")
REFERENCE_ELECTRICITY_FLOW_ID = "fc406690-160c-37d5-bf36-added9542164"
PROCESS_CATEGORY = (
    "22: Utilities / 2211: Electric Power Generation, Transmission and Distribution / International"
)

# Skip synthetic mixes by default. Keep OTHF and MIXED because they are
# represented as explicit resource categories in the baseline package.
SKIP_RESOURCES = {"ALL"}


def _load_processes_from_commons(repo_key: str = BASELINE_REPO_KEY):
    """Load all processes for a Commons repo via flcac-utils."""
    data = read_commons_data({repo_key: "PROCESS"}, auth=False)
    processes_by_id = {}
    processes_by_name = {}
    for process in data[repo_key]:
        process_dict = process.to_dict()
        process_id = process_dict.get("@id")
        process_name = process_dict.get("name")
        if process_id:
            processes_by_id[process_id] = process_dict
        if process_name:
            processes_by_name[process_name] = process_dict
    return processes_by_id, processes_by_name


def _provider_info(exchange):
    provider = exchange.get("defaultProvider", {}) if isinstance(exchange, dict) else {}
    return provider.get("@id"), provider.get("name")


def _find_reference_exchange(process):
    for exchange in process.get("exchanges", []):
        flow = exchange.get("flow", {}) if isinstance(exchange, dict) else {}
        provider_id, _ = _provider_info(exchange)
        if (
            str(flow.get("@id", "")) == REFERENCE_ELECTRICITY_FLOW_ID
            and provider_id is None
            and abs(float(exchange.get("amount", 0.0)) - 1.0) < 1e-12
        ):
            return exchange

    for exchange in process.get("exchanges", []):
        provider_id, _ = _provider_info(exchange)
        if provider_id is None and abs(float(exchange.get("amount", 0.0)) - 1.0) < 1e-12:
            return exchange

    raise ValueError(
        f"Could not identify reference exchange for process: {process.get('name')}"
    )


def _extract_us_ba_weights(us_grid_mix_process):
    weights = {}

    for exchange in us_grid_mix_process.get("exchanges", []):
        provider_id, provider_name = _provider_info(exchange)
        if not provider_id or not provider_name:
            continue
        if not provider_name.startswith(GRID_GENERATION_MIX_PREFIX):
            continue

        ba_name = provider_name[len(GRID_GENERATION_MIX_PREFIX) :].strip()
        weights[provider_id] = {
            "generation_mix_name": provider_name,
            "ba_name": ba_name,
            "weight": float(exchange.get("amount", 0.0)),
        }

    if not weights:
        raise ValueError(
            "No BA generation-mix provider exchanges found in US grid consumption mix."
        )

    total = sum(entry["weight"] for entry in weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"US BA weights do not sum to 1.0 (sum={total}).")

    return weights


def _resource_name_from_process(process_name: str):
    match = RESOURCE_PROCESS_PATTERN.match(process_name)
    if not match:
        return None
    return match.group(1).strip()


def _compute_resource_contributions(processes_by_id, us_ba_weights):
    resource_contribs = defaultdict(list)

    for generation_mix_id, ba_meta in us_ba_weights.items():
        generation_mix_process = processes_by_id.get(generation_mix_id)
        if generation_mix_process is None:
            continue

        ba_weight = ba_meta["weight"]
        ba_name = ba_meta["ba_name"]

        for exchange in generation_mix_process.get("exchanges", []):
            provider_id, provider_name = _provider_info(exchange)
            if not provider_id or not provider_name:
                continue

            resource_name = _resource_name_from_process(provider_name)
            if not resource_name or resource_name in SKIP_RESOURCES:
                continue

            resource_fraction = float(exchange.get("amount", 0.0))
            contribution = ba_weight * resource_fraction
            if contribution <= 0.0:
                continue

            resource_contribs[resource_name].append(
                {
                    "ba_name": ba_name,
                    "ba_weight": ba_weight,
                    "resource_fraction": resource_fraction,
                    "contribution": contribution,
                    "process_id": provider_id,
                    "process_name": provider_name,
                }
            )

    return resource_contribs


def _aggregate_resource_process(resource, contributors, processes_by_id, us_grid_mix_process):
    total_contribution = sum(c["contribution"] for c in contributors)
    if total_contribution <= 0.0:
        return None

    normalized = []
    for item in contributors:
        w = item["contribution"] / total_contribution
        normalized.append((w, item))

    # Base metadata from highest-weight contributor process.
    normalized.sort(key=lambda x: x[0], reverse=True)
    base_process = processes_by_id[normalized[0][1]["process_id"]]
    aggregated_process = copy.deepcopy(base_process)

    aggregated_process["@id"] = make_uuid(f"Electricity - {resource} - US average")
    aggregated_process["name"] = f"Electricity - {resource} - US average"

    old_description = str(aggregated_process.get("description", "")).strip()
    method_note = (
        "US-average resource process represented as a balancing-authority "
        "provider mix. Each input exchange links to a BA resource process and "
        "uses a normalized weight back-calculated from the process "
        f"'{US_GRID_CONSUMPTION_MIX_NAME}'."
    )
    aggregated_process["description"] = f"{old_description}\n\n{method_note}".strip()
    aggregated_process["lastChange"] = (
        datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    )

    if isinstance(us_grid_mix_process.get("location"), dict):
        aggregated_process["location"] = copy.deepcopy(us_grid_mix_process["location"])

    aggregated_process["category"] = PROCESS_CATEGORY

    reference_exchange = copy.deepcopy(_find_reference_exchange(base_process))
    reference_exchange["@id"] = make_uuid(
        aggregated_process["@id"], "resource_reference", resource
    )
    reference_exchange["amount"] = 1.0
    reference_exchange.pop("defaultProvider", None)
    reference_exchange["isInput"] = False
    reference_exchange["isAvoidedProduct"] = False
    reference_exchange["isQuantitativeReference"] = True
    reference_exchange["internalId"] = 1

    aggregated_exchanges = [reference_exchange]
    for idx, (weight, item) in enumerate(normalized, start=2):
        mix_exchange = copy.deepcopy(reference_exchange)
        mix_exchange["@id"] = make_uuid(
            aggregated_process["@id"],
            "resource_mix",
            resource,
            item["process_id"],
            str(idx),
        )
        mix_exchange["amount"] = weight
        mix_exchange["isInput"] = True
        mix_exchange["isAvoidedProduct"] = False
        mix_exchange["isQuantitativeReference"] = False
        mix_exchange["internalId"] = idx
        mix_exchange["defaultProvider"] = {
            "@type": "Process",
            "@id": item["process_id"],
            "name": item["process_name"],
        }
        aggregated_exchanges.append(mix_exchange)

    aggregated_process["exchanges"] = aggregated_exchanges
    aggregated_process["lastInternalId"] = len(aggregated_exchanges)

    return {
        "process": aggregated_process,
        "resource_share": total_contribution,
        "contributors": [
            {
                "resource": resource,
                "normalized_weight": weight,
                **item,
            }
            for weight, item in normalized
        ],
    }


def _write_audit_csvs(aggregated_results):
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

    summary_path = EXTRACT_DIR / "resource_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["resource", "us_share_from_mix", "contributors"])
        for resource in sorted(aggregated_results):
            result = aggregated_results[resource]
            writer.writerow(
                [
                    resource,
                    f"{result['resource_share']:.15f}",
                    len(result["contributors"]),
                ]
            )

    detail_path = EXTRACT_DIR / "resource_ba_weights.csv"
    with detail_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "resource",
                "ba_name",
                "generation_mix_process",
                "resource_process",
                "ba_weight_in_us_mix",
                "resource_fraction_in_ba_generation_mix",
                "us_contribution",
                "normalized_weight_in_resource_average",
            ]
        )
        for resource in sorted(aggregated_results):
            for row in aggregated_results[resource]["contributors"]:
                writer.writerow(
                    [
                        resource,
                        row["ba_name"],
                        f"{GRID_GENERATION_MIX_PREFIX}{row['ba_name']}",
                        row["process_name"],
                        f"{row['ba_weight']:.15f}",
                        f"{row['resource_fraction']:.15f}",
                        f"{row['contribution']:.15f}",
                        f"{row['normalized_weight']:.15f}",
                    ]
                )

    return summary_path, detail_path


def _write_olca_package(processes_for_package):
    """Write JSON-LD zip via flcac-utils and extract with extract_latest_zip."""
    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    processes = {
        str(process["@id"]): olca.Process.from_dict(process)
        for process in processes_for_package
    }
    write_objects(
        "us_fuel_generation_mixes",
        {},
        [],
        processes,
        out_path=ZIP_DIR,
    )
    zip_path = max(
        ZIP_DIR.glob("us_fuel_generation_mixes_olca2.0_*.zip"),
        key=lambda p: p.stat().st_mtime,
    )
    if EXTRACT_DIR.exists():
        shutil.rmtree(EXTRACT_DIR)
    return extract_latest_zip(
        zip_path,
        BASE_DIR,
        output_folder_name=Path("output") / "us_fuel_generation_mixes",
    )


def main():
    processes_by_id, processes_by_name = _load_processes_from_commons()
    print(f"Loaded {len(processes_by_id)} processes from {BASELINE_REPO_KEY}")

    us_grid_mix = processes_by_name.get(US_GRID_CONSUMPTION_MIX_NAME)
    if us_grid_mix is None:
        raise ValueError(f"Missing process: {US_GRID_CONSUMPTION_MIX_NAME}")

    us_ba_weights = _extract_us_ba_weights(us_grid_mix)
    resource_contribs = _compute_resource_contributions(processes_by_id, us_ba_weights)

    aggregated_results = {}
    for resource, contributors in sorted(resource_contribs.items()):
        aggregated = _aggregate_resource_process(
            resource=resource,
            contributors=contributors,
            processes_by_id=processes_by_id,
            us_grid_mix_process=us_grid_mix,
        )
        if aggregated is not None:
            aggregated_results[resource] = aggregated

    if not aggregated_results:
        raise ValueError("No resource aggregates were generated.")

    package_dir = _write_olca_package(
        [result["process"] for result in aggregated_results.values()]
    )
    summary_path, detail_path = _write_audit_csvs(aggregated_results)

    print(f"Generated {len(aggregated_results)} US-average resource processes.")
    print(f"Zip directory: {ZIP_DIR}")
    print(f"Summary CSV: {summary_path}")
    print(f"Detailed BA weights CSV: {detail_path}")
    print(f"Extracted JSON-LD: {package_dir}")


if __name__ == "__main__":
    main()
