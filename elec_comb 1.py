#!/usr/bin/env python3
"""
Aggregate balancing-authority electricity resource processes into US-average
resource processes (SOLAR, WIND, COAL, etc.) using the US grid consumption mix.

Method summary
1) Read BA weights from:
   Electricity; at grid; consumption mix - US - US
2) For each BA generation-mix provider in that US process, read the BA
   generation mix shares by resource process (e.g., Electricity - SOLAR - BA).
3) Compute resource contribution weights:
   contribution(BA, resource) = US_BA_weight * BA_resource_fraction
4) Normalize contribution weights within each resource so the selected BA
   resource providers sum to 1.0 for that resource.
5) Create one aggregated process per resource by weighted averaging all
   non-reference exchanges from BA resource processes.

Outputs (created in ./us_avg_resource_processes):
- process_<RESOURCE>.json: openLCA process object for US-average resource mix
- resource_summary.csv: resource shares and contributor counts
- resource_ba_weights.csv: detailed BA weights per resource
"""

from __future__ import annotations

import copy
import csv
import json
import re
import uuid
import zipfile
from collections import defaultdict
import datetime
from pathlib import Path
from typing import Dict, List, Tuple


BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
ZIP_PATH = BASE_DIR / "Federal_LCA_Commons-US_electricity_baseline.zip"
OUTPUTS_ROOT = PROJECT_ROOT / "outputs"
OUTPUT_DIR = OUTPUTS_ROOT / "us_avg_resource_processes"
OPENLCA_PACKAGE_DIR = OUTPUT_DIR / "openlca_package"
OPENLCA_PACKAGE_ZIP = OUTPUT_DIR / "openlca_package.zip"
ELECTRICITY_CSV_PATH = BASE_DIR / "electricity.csv"

US_GRID_CONSUMPTION_MIX_NAME = "Electricity; at grid; consumption mix - US - US"
US_USER_CONSUMPTION_MIX_NAME = "Electricity; at user; consumption mix - US - US"
GRID_GENERATION_MIX_PREFIX = "Electricity; at grid; generation mix - "
RESOURCE_PROCESS_PATTERN = re.compile(r"^Electricity - ([^-]+?) - (.+)$")
REFERENCE_ELECTRICITY_FLOW_ID = "fc406690-160c-37d5-bf36-added9542164"

# Skip synthetic mixes by default. Keep OTHF and MIXED because they are
# represented as explicit resource categories in the baseline package.
SKIP_RESOURCES = {"ALL"}

# Mapping from electricity.csv parameter names to generated US-average resources.
CSV_PARAM_TO_RESOURCE = {
	"elec_nat_gas%": "GAS",
	"elec_oil%": "OIL",
	"elec_coal%": "COAL",
	"elec_nuclear%": "NUCLEAR",
	"elec_hydro%": "HYDRO",
	"elec_wind%": "WIND",
	"elec_solar%": "SOLAR",
	"elec_biomass%": "BIOMASS",
	"elec_geothermal%": "GEOTHERMAL",
	"elec_other%": "OTHF",
}

USA_MIRROR_REGIONS = {"USA", "US"}


def _stable_uuid(*parts):
	seed = "::".join(str(part) for part in parts)
	return str(uuid.uuid5(uuid.NAMESPACE_URL, f"swolf-elec-comb::{seed}"))


def _stable_exchange_id(process_id: str, role: str, index: int, *parts):
	return _stable_uuid("exchange", process_id, role, index, *parts)


def _load_processes_from_zip(zip_path: Path):
	processes_by_id = {}
	processes_by_name = {}

	with zipfile.ZipFile(zip_path, "r") as archive:
		for member in archive.namelist():
			if not member.startswith("processes/") or not member.endswith(".json"):
				continue
			process = json.loads(archive.read(member))
			process_id = process.get("@id")
			process_name = process.get("name")
			if process_id:
				processes_by_id[process_id] = process
			if process_name:
				processes_by_name[process_name] = process

	return processes_by_id, processes_by_name


def _provider_info(exchange):
	provider = exchange.get("defaultProvider", {}) if isinstance(exchange, dict) else {}
	provider_id = provider.get("@id")
	provider_name = provider.get("name")
	return provider_id, provider_name


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

	# Fallback: first exchange with amount=1 and no provider.
	for exchange in process.get("exchanges", []):
		provider_id, _ = _provider_info(exchange)
		if provider_id is None and abs(float(exchange.get("amount", 0.0)) - 1.0) < 1e-12:
			return exchange

	raise ValueError(f"Could not identify reference exchange for process: {process.get('name')}")


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
		raise ValueError("No BA generation-mix provider exchanges found in US grid consumption mix.")

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
	# resource -> list of contributors
	# contributor fields: ba_name, ba_weight, resource_fraction, contribution,
	# process_id, process_name
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


def _exchange_signature(exchange):
	flow = exchange.get("flow", {})
	unit = exchange.get("unit", {})
	flow_property = exchange.get("flowProperty", {})
	provider = exchange.get("defaultProvider", {})
	return (
		str(flow.get("@id", "")),
		str(provider.get("@id", "")),
		str(unit.get("@id", "")),
		str(flow_property.get("@id", "")),
		bool(exchange.get("input", False)),
		bool(exchange.get("avoidedProduct", False)),
	)


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

	aggregated_process["@id"] = str(
		uuid.uuid5(
			uuid.NAMESPACE_URL,
			f"swolf-us-avg-electricity-resource::{resource}",
		)
	)
	aggregated_process["name"] = f"Electricity - {resource} - US average"

	old_description = str(aggregated_process.get("description", "")).strip()
	method_note = (
		"US-average resource process represented as a balancing-authority "
		"provider mix. Each input exchange links to a BA resource process and "
		"uses a normalized weight back-calculated from the process "
		f"'{US_GRID_CONSUMPTION_MIX_NAME}'."
	)
	aggregated_process["description"] = f"{old_description}\n\n{method_note}".strip()
	aggregated_process["lastChange"] = "2026-08-06T00:00:00.000Z"

	if isinstance(us_grid_mix_process.get("location"), dict):
		aggregated_process["location"] = copy.deepcopy(us_grid_mix_process["location"])

	# Keep category style, but pin to US average variant.
	category = str(aggregated_process.get("category", "")).strip()
	if category:
		parts = category.split("/")
		if parts:
			base_cat = parts[0]
			aggregated_process["category"] = f"{base_cat}/US_average/{resource}"

	reference_exchange = copy.deepcopy(_find_reference_exchange(base_process))
	reference_exchange["@id"] = _stable_exchange_id(
		aggregated_process["@id"],
		"resource_reference",
		1,
		resource,
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
		mix_exchange["@id"] = _stable_exchange_id(
			aggregated_process["@id"],
			"resource_mix",
			idx,
			resource,
			item["process_id"],
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


def _write_outputs(aggregated_results):
	OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

	for resource, result in aggregated_results.items():
		process_path = OUTPUT_DIR / f"process_{resource}.json"
		with process_path.open("w", encoding="utf-8") as handle:
			json.dump(result["process"], handle, indent=2)

	summary_path = OUTPUT_DIR / "resource_summary.csv"
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

	detail_path = OUTPUT_DIR / "resource_ba_weights.csv"
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


def _safe_name(value):
	return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(value))


def _sanitize_parameter_name(value: str):
	name = str(value).strip().replace("%", "pct")
	name = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
	while "__" in name:
		name = name.replace("__", "_")
	return name.strip("_") or "param"


def _parse_percent_value(value):
	text = str(value).strip()
	if text.endswith("%"):
		text = text[:-1].strip()
	if not text:
		return 0.0
	return float(text) / 100.0


def _read_electricity_csv_inputs(aggregated_results):
	if not ELECTRICITY_CSV_PATH.exists():
		raise FileNotFoundError(f"electricity.csv not found: {ELECTRICITY_CSV_PATH}")

	provider_by_resource = {}
	for resource, result in aggregated_results.items():
		process = result["process"]
		provider_by_resource[resource] = {
			"@id": process.get("@id"),
			"name": process.get("name"),
		}

	region_resource_rows = defaultdict(list)
	region_grid_loss = {}
	with ELECTRICITY_CSV_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			param = str(row.get("Parameter", "")).strip()
			region = str(row.get("Region", "")).strip()
			if not region:
				continue

			parameter_key = str(row.get("Parameter + Region", "")).strip() or f"{param}_{region}"
			value_fraction = _parse_percent_value(row.get("Standard Value", "0"))

			if param == "grid_loss%":
				if value_fraction < 0 or value_fraction >= 1:
					raise ValueError(
						f"grid_loss% must be in [0, 100) for {region}, got {value_fraction * 100:.6f}%"
					)
				region_grid_loss[region] = {
					"parameter_key": parameter_key,
					"loss_fraction": value_fraction,
				}
				continue

			resource = CSV_PARAM_TO_RESOURCE.get(param)
			if resource is None or resource not in provider_by_resource:
				continue

			if value_fraction < 0:
				raise ValueError(
					f"Negative electricity share for {region} / {param}: {value_fraction}"
				)

			region_resource_rows[region].append(
				{
					"resource": resource,
					"parameter_key": parameter_key,
					"share_fraction": value_fraction,
				}
			)

	return provider_by_resource, region_resource_rows, region_grid_loss


def _build_parameterized_region_generation_mix_processes(aggregated_results, us_grid_mix_process):
	provider_by_resource, region_resource_rows, region_grid_loss = _read_electricity_csv_inputs(
		aggregated_results
	)

	results = {}
	for region, rows in sorted(region_resource_rows.items()):
		mirror_usa_baseline = region in USA_MIRROR_REGIONS
		process = copy.deepcopy(us_grid_mix_process)
		process["@id"] = str(
			uuid.uuid5(
				uuid.NAMESPACE_URL,
				f"swolf-regional-electricity-generation-mix::{region}",
			)
		)
		process["name"] = f"Electricity; at grid; generation mix - {region} - parameterized"

		old_description = str(process.get("description", "")).strip()
		if mirror_usa_baseline:
			process["description"] = (
				f"{old_description}\n\n"
				"USA at-grid generation mix mirrors the baseline US balancing-authority "
				"provider structure to preserve comparability with ElectricityLCI baseline "
				"results."
			).strip()
		else:
			process["description"] = (
				f"{old_description}\n\n"
				"Parameterized regional at-grid generation mix built from electricity.csv. "
				"Input exchanges are linked to US-average resource processes and use "
				"process parameters for each resource share percentage."
			).strip()
		process["lastChange"] = "2026-08-06T00:00:00.000Z"

		category = str(process.get("category", "")).strip()
		if category:
			parts = category.split("/")
			base_cat = parts[0]
			process["category"] = f"{base_cat}/Regional_parameterized/{_safe_name(region)}/at_grid"

		if mirror_usa_baseline:
			for idx, exchange in enumerate(process.get("exchanges", []), start=1):
				provider_id, provider_name = _provider_info(exchange)
				flow_id = str((exchange.get("flow") or {}).get("@id", ""))
				exchange["@id"] = _stable_exchange_id(
					process["@id"],
					"generation_mirror",
					idx,
					flow_id,
					provider_id or "none",
					provider_name or "none",
				)
			process["parameters"] = []
			process["lastInternalId"] = len(process.get("exchanges", []))
		else:
			reference_exchange = copy.deepcopy(_find_reference_exchange(us_grid_mix_process))
			reference_exchange["@id"] = _stable_exchange_id(
				process["@id"],
				"generation_reference",
				1,
				region,
			)
			reference_exchange["amount"] = 1.0
			reference_exchange["isInput"] = False
			reference_exchange["isAvoidedProduct"] = False
			reference_exchange["isQuantitativeReference"] = True
			reference_exchange["internalId"] = 1
			reference_exchange.pop("defaultProvider", None)

			exchanges = [reference_exchange]
			parameters = []
			nonzero_rows = [row for row in rows if row["share_fraction"] > 0]
			nonzero_rows.sort(key=lambda row: row["share_fraction"], reverse=True)

			for idx, row in enumerate(nonzero_rows, start=2):
				resource = row["resource"]
				provider = provider_by_resource[resource]
				if not provider.get("@id") or not provider.get("name"):
					continue

				parameter_name = _sanitize_parameter_name(row["parameter_key"])
				parameter_percent = row["share_fraction"] * 100.0
				parameters.append(
					{
						"@type": "Parameter",
						"@id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"{process['@id']}::{parameter_name}")),
						"name": parameter_name,
						"description": f"From electricity.csv: {row['parameter_key']}",
						"inputParameter": True,
						"value": parameter_percent,
					}
				)

				exchange = copy.deepcopy(reference_exchange)
				exchange["@id"] = _stable_exchange_id(
					process["@id"],
					"generation_input",
					idx,
					region,
					resource,
					provider["@id"],
				)
				exchange["isInput"] = True
				exchange["isQuantitativeReference"] = False
				exchange["internalId"] = idx
				exchange["amount"] = row["share_fraction"]
				exchange["amountFormula"] = f"{parameter_name} / 100"
				exchange["defaultProvider"] = {
					"@type": "Process",
					"@id": provider["@id"],
					"name": provider["name"],
				}
				exchanges.append(exchange)

			process["exchanges"] = exchanges
			process["lastInternalId"] = len(exchanges)
			process["parameters"] = parameters

		results[region] = {
			"process": process,
			"resource_rows": rows,
			"grid_loss": region_grid_loss.get(
				region, {"parameter_key": f"grid_loss%_{region}", "loss_fraction": 0.0}
			),
		}

	return results


def _build_parameterized_region_consumption_mix_processes(
	region_generation_results, us_user_mix_process
):
	# Find a canonical high-voltage input template from the baseline US at-user
	# consumption mix process.
	input_template = None
	for exchange in us_user_mix_process.get("exchanges", []):
		provider_id, _ = _provider_info(exchange)
		flow_id = str((exchange.get("flow") or {}).get("@id", ""))
		if provider_id and flow_id == REFERENCE_ELECTRICITY_FLOW_ID:
			input_template = exchange
			break
	if input_template is None:
		for exchange in us_user_mix_process.get("exchanges", []):
			provider_id, _ = _provider_info(exchange)
			if provider_id:
				input_template = exchange
				break
	if input_template is None:
		raise ValueError(
			"Could not identify high-voltage provider input exchange in US user consumption mix process."
		)

	results = {}
	for region, generation_data in sorted(region_generation_results.items()):
		mirror_usa_baseline = region in USA_MIRROR_REGIONS
		generation_process = generation_data["process"]
		grid_loss_info = generation_data.get("grid_loss", {})
		loss_fraction = float(grid_loss_info.get("loss_fraction", 0.0))
		loss_parameter_key = str(grid_loss_info.get("parameter_key", f"grid_loss%_{region}"))
		if loss_fraction < 0 or loss_fraction >= 1:
			raise ValueError(
				f"grid_loss% must be in [0, 100) for {region}, got {loss_fraction * 100:.6f}%"
			)

		process = copy.deepcopy(us_user_mix_process)
		process["@id"] = str(
			uuid.uuid5(
				uuid.NAMESPACE_URL,
				f"swolf-regional-electricity-consumption-mix::{region}",
			)
		)
		process["name"] = f"Electricity; at user; consumption mix - {region} - parameterized"

		old_description = str(process.get("description", "")).strip()
		if mirror_usa_baseline:
			process["description"] = (
				f"{old_description}\n\n"
				"USA at-user consumption mix mirrors the baseline US structure and relinks "
				"its high-voltage provider to the generated USA at-grid mix for consistency."
			).strip()
		else:
			process["description"] = (
				f"{old_description}\n\n"
				"Parameterized regional at-user consumption mix built from electricity.csv. "
				"The high-voltage electricity input is linked to the regional at-grid "
				"generation mix and scaled by grid_loss%."
			).strip()
		process["lastChange"] = "2026-08-06T00:00:00.000Z"

		category = str(process.get("category", "")).strip()
		if category:
			parts = category.split("/")
			base_cat = parts[0]
			process["category"] = f"{base_cat}/Regional_parameterized/{_safe_name(region)}/at_user"

		if mirror_usa_baseline:
			exchanges = process.get("exchanges", [])
			hv_index = None
			for idx, exchange in enumerate(exchanges):
				provider_id, _ = _provider_info(exchange)
				flow_id = str((exchange.get("flow") or {}).get("@id", ""))
				if provider_id and flow_id == REFERENCE_ELECTRICITY_FLOW_ID:
					hv_index = idx
					break
			if hv_index is None:
				for idx, exchange in enumerate(exchanges):
					provider_id, _ = _provider_info(exchange)
					if provider_id:
						hv_index = idx
						break

			if hv_index is not None:
				exchanges[hv_index]["defaultProvider"] = {
					"@type": "Process",
					"@id": generation_process.get("@id"),
					"name": generation_process.get("name"),
				}

			for idx, exchange in enumerate(exchanges, start=1):
				provider_id, provider_name = _provider_info(exchange)
				flow_id = str((exchange.get("flow") or {}).get("@id", ""))
				exchange["@id"] = _stable_exchange_id(
					process["@id"],
					"consumption_mirror",
					idx,
					flow_id,
					provider_id or "none",
					provider_name or "none",
				)
			process["parameters"] = []
			process["lastInternalId"] = len(exchanges)
		else:
			reference_exchange = copy.deepcopy(_find_reference_exchange(us_user_mix_process))
			reference_exchange["@id"] = _stable_exchange_id(
				process["@id"],
				"consumption_reference",
				1,
				region,
			)
			reference_exchange["amount"] = 1.0
			reference_exchange["isInput"] = False
			reference_exchange["isAvoidedProduct"] = False
			reference_exchange["isQuantitativeReference"] = True
			reference_exchange["internalId"] = 1
			reference_exchange.pop("defaultProvider", None)

			loss_parameter_name = _sanitize_parameter_name(loss_parameter_key)
			loss_percent = loss_fraction * 100.0
			parameters = [
				{
					"@type": "Parameter",
					"@id": str(
						uuid.uuid5(uuid.NAMESPACE_URL, f"{process['@id']}::{loss_parameter_name}")
					),
					"name": loss_parameter_name,
					"description": f"From electricity.csv: {loss_parameter_key}",
					"inputParameter": True,
					"value": loss_percent,
				}
			]

			input_exchange = copy.deepcopy(input_template)
			input_exchange["@id"] = _stable_exchange_id(
				process["@id"],
				"consumption_input",
				2,
				region,
				generation_process.get("@id", ""),
			)
			input_exchange["isInput"] = True
			input_exchange["isQuantitativeReference"] = False
			input_exchange["isAvoidedProduct"] = False
			input_exchange["internalId"] = 2
			input_exchange["amount"] = 1.0 / (1.0 - loss_fraction)
			input_exchange["amountFormula"] = f"1 / (1 - {loss_parameter_name} / 100)"
			input_exchange["defaultProvider"] = {
				"@type": "Process",
				"@id": generation_process.get("@id"),
				"name": generation_process.get("name"),
			}

			process["exchanges"] = [reference_exchange, input_exchange]
			process["lastInternalId"] = 2
			process["parameters"] = parameters

		results[region] = {
			"process": process,
			"grid_loss": grid_loss_info,
		}

	return results


def _write_parameterized_region_outputs(region_generation_results, region_consumption_results):
	for region in sorted(region_generation_results):
		process = region_generation_results[region]["process"]
		process_path = OUTPUT_DIR / f"process_region_generation_{_safe_name(region)}.json"
		with process_path.open("w", encoding="utf-8") as handle:
			json.dump(process, handle, indent=2)

	for region in sorted(region_consumption_results):
		process = region_consumption_results[region]["process"]
		process_path = OUTPUT_DIR / f"process_region_consumption_{_safe_name(region)}.json"
		with process_path.open("w", encoding="utf-8") as handle:
			json.dump(process, handle, indent=2)

	region_summary_path = OUTPUT_DIR / "region_mix_summary.csv"
	with region_summary_path.open("w", newline="", encoding="utf-8") as handle:
		writer = csv.writer(handle)
		writer.writerow(
			[
				"region",
				"resource",
				"parameter_key",
				"share_fraction",
				"share_percent",
				"grid_loss_fraction",
				"grid_loss_percent",
			]
		)
		for region in sorted(region_generation_results):
			grid_loss_fraction = float(
				region_generation_results[region].get("grid_loss", {}).get("loss_fraction", 0.0)
			)
			for row in sorted(
				region_generation_results[region]["resource_rows"], key=lambda r: r["resource"]
			):
				writer.writerow(
					[
						region,
						row["resource"],
						row["parameter_key"],
						f"{row['share_fraction']:.15f}",
						f"{row['share_fraction'] * 100.0:.6f}",
						f"{grid_loss_fraction:.15f}",
						f"{grid_loss_fraction * 100.0:.6f}",
					]
				)

	return region_summary_path


def _write_openlca_package(processes_for_package):
	# Build a baseline-compatible openLCA package by reusing all non-process
	# dataset elements from the baseline archive and replacing process datasets
	# with the aggregated US-average resource processes.
	if OPENLCA_PACKAGE_DIR.exists():
		for existing in sorted(OPENLCA_PACKAGE_DIR.rglob("*"), reverse=True):
			if existing.is_file():
				existing.unlink()
			elif existing.is_dir():
				existing.rmdir()

	OPENLCA_PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
	(OPENLCA_PACKAGE_DIR / "processes").mkdir(parents=True, exist_ok=True)

	with zipfile.ZipFile(ZIP_PATH, "r") as archive:
		for member in archive.namelist():
			if member.startswith("processes/"):
				continue

			target = OPENLCA_PACKAGE_DIR / Path(member)
			if member.endswith("/"):
				target.mkdir(parents=True, exist_ok=True)
				continue

			target.parent.mkdir(parents=True, exist_ok=True)
			with target.open("wb") as handle:
				handle.write(archive.read(member))

	for process in sorted(processes_for_package, key=lambda proc: str(proc.get("name", ""))):
		process_id = str(process.get("@id", "")).strip()
		if not process_id:
			process_name = str(process.get("name", "")).strip() or "unnamed_process"
			process_id = _stable_uuid("process", process_name)
			process["@id"] = process_id

		file_name = f"{process_id}.json"
		process_path = OPENLCA_PACKAGE_DIR / "processes" / file_name
		with process_path.open("w", encoding="utf-8") as handle:
			json.dump(process, handle, indent=2)

	if OPENLCA_PACKAGE_ZIP.exists():
		OPENLCA_PACKAGE_ZIP.unlink()

	with zipfile.ZipFile(OPENLCA_PACKAGE_ZIP, "w", zipfile.ZIP_DEFLATED) as archive:
		for path in sorted(OPENLCA_PACKAGE_DIR.rglob("*")):
			archive_path = path.relative_to(OPENLCA_PACKAGE_DIR).as_posix()
			if path.is_dir():
				archive.write(path, f"{archive_path}/")
			elif path.is_file():
				archive.write(path, archive_path)

	return OPENLCA_PACKAGE_DIR, OPENLCA_PACKAGE_ZIP


def main():
	if not ZIP_PATH.exists():
		raise FileNotFoundError(f"Baseline zip not found: {ZIP_PATH}")

	processes_by_id, processes_by_name = _load_processes_from_zip(ZIP_PATH)

	us_grid_mix = processes_by_name.get(US_GRID_CONSUMPTION_MIX_NAME)
	if us_grid_mix is None:
		raise ValueError(f"Missing process: {US_GRID_CONSUMPTION_MIX_NAME}")

	us_user_mix = processes_by_name.get(US_USER_CONSUMPTION_MIX_NAME)
	if us_user_mix is None:
		raise ValueError(f"Missing process: {US_USER_CONSUMPTION_MIX_NAME}")

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

	region_generation_results = _build_parameterized_region_generation_mix_processes(
		aggregated_results=aggregated_results,
		us_grid_mix_process=us_grid_mix,
	)
	region_consumption_results = _build_parameterized_region_consumption_mix_processes(
		region_generation_results=region_generation_results,
		us_user_mix_process=us_user_mix,
	)

	summary_path, detail_path = _write_outputs(aggregated_results)
	region_summary_path = _write_parameterized_region_outputs(
		region_generation_results=region_generation_results,
		region_consumption_results=region_consumption_results,
	)

	processes_for_package = [
		result["process"] for result in aggregated_results.values()
	] + [result["process"] for result in region_generation_results.values()] + [
		result["process"] for result in region_consumption_results.values()
	]
	package_dir, package_zip = _write_openlca_package(processes_for_package)

	print(f"Generated {len(aggregated_results)} US-average resource processes.")
	print(
		f"Generated {len(region_generation_results)} parameterized regional at-grid generation mix processes."
	)
	print(
		f"Generated {len(region_consumption_results)} parameterized regional at-user consumption mix processes."
	)
	print(f"Output directory: {OUTPUT_DIR}")
	print(f"Summary CSV: {summary_path}")
	print(f"Detailed BA weights CSV: {detail_path}")
	print(f"Regional mix summary CSV: {region_summary_path}")
	print(f"openLCA package directory: {package_dir}")
	print(f"openLCA package zip: {package_zip}")


if __name__ == "__main__":
	main()
