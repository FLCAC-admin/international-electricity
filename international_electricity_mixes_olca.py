"""
Generate processes of international electricity mixes
"""

import copy
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import sys
from esupy.util import make_uuid

parent_path = Path(__file__).parent
data_path = parent_path / 'data'
out_path = parent_path / 'output'

with open(data_path / 'electricity.yaml') as f:
    meta = yaml.safe_load(f)
years = meta['Years']

# ember data
# https://ember-climate.org/data-catalogue/yearly-electricity-data/
# data_url = 'https://ember-climate.org/app/uploads/2022/07/yearly_full_release_long_format.csv'
# getting a Forbidden error when trying to access via url
# alternate use API: https://api.ember-energy.org/docs
data_csv = data_path / 'yearly_full_release_long_format.csv'
try:
    df_orig = pd.read_csv(data_csv)
except FileNotFoundError:
    raise FileNotFoundError("Electricity data file must be downloaded and saved to "
                            "the electricity folder to proceed. \n"
                            "See https://ember-climate.org/data-catalogue/yearly-electricity-data/")

# Prepare dataframe of electricity shares
df = (df_orig
      .query('Year.isin(@years)')
      .query('`Area type` == "Country or economy"')
      .query('Category == "Electricity generation"')
      .query('Subcategory == "Fuel"')
      .query('Unit == "TWh"')
      .filter(['Year', 'Area', 'Country', 'ISO 3 code',
               'Variable', 'Unit', 'Value'])
      .rename(columns={'ISO 3 code': 'CountryCode',
                       'Variable': 'Fuel'})
      )
df['share'] = df['Value'] / df.groupby(['CountryCode', 'Year'])['Value'].transform('sum')

countries = (df[['Area', 'CountryCode', 'Year']]
             .sort_values(by='Year')
             .drop_duplicates(subset=['Area', 'CountryCode'], keep='last')
             )

# Write to markdown
## TODO: note a few countries get dropped, maybe missing location objects?
markdown_file = countries.sort_values(by='Area').to_markdown(index=False)
with open(parent_path / "country_list.md", "w") as f:
    f.write("# Country List\n\n" + markdown_file)

# merge back in to keep only the latest set of data for each area
df = df.merge(countries, how='inner')
df = df.dropna(subset='share')

## Round the values to N digits but make sure they still sum to 1
def round_group(group, value_col, digits=4):
    factor = 10 ** digits
    shares = group[value_col].values
    scaled = shares * factor
    floored = np.floor(scaled)
    remainder = int(factor - floored.sum())

    # Distribute the remainder to the largest fractional parts
    fractional_parts = scaled - floored
    indices = np.argsort(-fractional_parts)[:remainder]
    floored[indices] += 1

    # Return the adjusted shares
    group[value_col] = floored / factor
    return group
df = (df.groupby("CountryCode", group_keys=False)
      [df.columns].apply(round_group, value_col='share'))

## As needed, write the fuels to a pivot
(df.pivot_table(index=['Year', 'Area', 'CountryCode', 'Unit'],
                         columns=['Fuel'],
                         values='Value', aggfunc='sum')
             .to_csv('fuel_by_country.csv'))


#%% Link to fuel specific flows in eLCI and prepare dataframe for oLCA
df_olca = pd.concat([(df
                      .assign(reference = False)
                      .assign(IsInput = True)),
                     (df[['Area', 'CountryCode', 'Year']]
                      .drop_duplicates()
                      .assign(reference = True)
                      .assign(share = 1)
                      .assign(IsInput = False)
                      )], ignore_index=True)
df_olca = (df_olca
           .assign(ProcessName = meta['Process']['ProcessName'])
           .assign(ProcessCategory = meta['Process']['ProcessCategory'])
           .assign(amount = df_olca['share'])
           .assign(unit = meta['Process']['Unit'])
           .assign(FlowName = meta['Process']['FlowName'])
           .assign(FlowUUID = meta['Process']['FlowUUID'])
           .assign(Context = meta['Process']['FlowContext'])
           # .assign(reference = df_olca['reference'].astype(bool).fillna(False))
           # .assign(IsInput = df_olca['IsInput'].astype(bool).fillna(True))
           .assign(FlowType = 'PRODUCT_FLOW')
           .rename(columns={'Fuel': 'description'})
           .assign(description = lambda x: x['description'].fillna(''))
           .drop(columns=['Unit', 'share', 'Value'])
           .query('CountryCode != "USA"') # Drop US data
           .reset_index(drop=True)
           )
# Update process name
df_olca['ProcessName'] = df_olca.apply(
    lambda row: row['ProcessName'].replace("<location>", row["Area"]), axis=1)

# Apply default providers based on fuel name
df_olca['default_provider_name'] = df_olca['description'].map(
    {k: v['ProcessName'] for k, v in meta['Fuel'].items()})
# convert default_provider_name to default_provider (UUID) in order to actually link them
# eLCI generated UUIDs don't seem to align with those based on creating from the name
# so need to pull them in manually
# df_olca['default_provider'] = df_olca.apply(
#     lambda row: np.nan if pd.isnull(row['default_provider_name']) else
#     make_uuid(row['default_provider_name']), axis=1)
df_olca['default_provider'] = df_olca['description'].map(
    {k: v['UUID'] for k, v in meta['Fuel'].items()})

## Confirm that totals are divisible by 1 (aka no missing data in gen mix)
if(df_olca['amount'].sum() % 1 != 0):
    print("WARNING: Check gen mix - not summing to 1!!")
    sys.exit()

#%% Assign exchange dqi
from flcac_utils.util import format_dqi_score, increment_dqi_value
df_olca['exchange_dqi'] = format_dqi_score(meta['DQI']['Flow'])
# update Technological correlation (position 4)
df_olca['exchange_dqi'] = np.where(
    df_olca['description'].isin(['Other Renewables', 'Other Fossil']),
    df_olca['exchange_dqi'].apply(lambda x: increment_dqi_value(x, 4)),
    df_olca['exchange_dqi'])
# drop DQI entry for reference flow
df_olca['exchange_dqi'] = np.where(df_olca['reference'] == True,
                                   '', df_olca['exchange_dqi'])

#%% Assign locations to processes
from flcac_utils.util import generate_locations_from_exchange_df
from esupy.location import read_iso_3166

df_olca = df_olca.merge(read_iso_3166()
                            .filter(['ISO-2d', 'ISO-3d'])
                            .rename(columns={'ISO-3d': 'CountryCode',
                                             'ISO-2d': 'location'}),
                        how='left')
locations = generate_locations_from_exchange_df(df_olca)


#%% Build supporting objects
from flcac_utils.generate_processes import build_location_dict
from flcac_utils.util import extract_actors_from_process_meta, \
    extract_sources_from_process_meta, extract_dqsystems

with open(data_path / 'electricity_process_metadata.yaml') as f:
    meta_docs = yaml.safe_load(f)

grid_meta = meta_docs['AtGrid']
user_meta_base = meta_docs['AtUser']
# per-country text templates; keyed by loss_source ('country' or 'region')
user_templates = user_meta_base.pop('templates')
(grid_meta, source_objs) = extract_sources_from_process_meta(
    grid_meta, bib_path=data_path / 'electricity.bib')
(user_meta_base, source_objs_u) = extract_sources_from_process_meta(
    user_meta_base, bib_path=data_path / 'electricity.bib')
source_objs.update(source_objs_u)
(grid_meta, actor_objs) = extract_actors_from_process_meta(grid_meta)
(user_meta_base, actor_objs_u) = extract_actors_from_process_meta(user_meta_base)
actor_objs.update(actor_objs_u)
dq_objs = extract_dqsystems(meta['DQI']['dqSystem'])
grid_meta['dq_entry'] = format_dqi_score(meta['DQI']['Process'])
user_meta_base['dq_entry'] = format_dqi_score(meta['DQI']['Process'])

# generate dictionary of location objects
location_objs = build_location_dict(df_olca, locations)
for loc in location_objs.values():
    # openLCA already roots locations under a Locations folder; use Country only.
    loc.category = "Country"

#%% Create json file
from flcac_utils.generate_processes import build_flow_dict, \
    build_process_dict, write_objects, validate_exchange_data
from flcac_utils.util import assign_year_to_meta
from flcac_utils.commons_api import get_single_object

validate_exchange_data(df_olca)
flows, new_flows = build_flow_dict(df_olca)
# Replace local stub with Commons metadata; still write this flow into the zip
ref_flow = get_single_object(
    "US Electricity Baseline", "FLOW", meta["Process"]["FlowUUID"])
flows[ref_flow.id] = ref_flow
if ref_flow.id not in new_flows:
    new_flows.append(ref_flow.id)
processes = {}
for year in df_olca.Year.unique():
    p_dict = build_process_dict(
        df_olca.query('Year == @year'),
        flows,
        meta=assign_year_to_meta(copy.deepcopy(grid_meta), int(year)),
        loc_objs=location_objs,
        source_objs=source_objs,
        actor_objs=actor_objs,
        dq_objs=dq_objs,
        )
    processes.update(p_dict)

#%% At-user generation mixes (T&D gross-up)
wb_csv = data_path / 'eg_elc_loss_zs.csv'
if not wb_csv.exists():
    raise FileNotFoundError(
        f"Missing {wb_csv}. Run download_worldbank_td_losses.py first.")
wb = pd.read_csv(wb_csv)
country_wb = wb.query('series_type == "country"')
region_wb = wb.query('series_type == "geographic_region"')
iso_region = (wb.query('series_type == "iso3_region"')
              .drop_duplicates('iso3').set_index('iso3'))
at_user = meta['AtUser']
base_dqi = format_dqi_score(meta['DQI']['Flow'])

grid_countries = (df_olca[['Area', 'CountryCode', 'Year', 'location']]
                  .drop_duplicates())
audit_rows, user_rows, user_info = [], [], {}
for _, c in grid_countries.iterrows():
    iso3, area, y_mix, loc = c.CountryCode, c.Area, int(c.Year), c.location
    rec = dict(CountryCode=iso3, Area=area, mix_year=y_mix, loss_year=None,
               loss_source='missing_wb_loss', loss_geo_name='', loss_geo_code='',
               L=None)
    hit = country_wb[(country_wb.iso3 == iso3) & (country_wb.year <= y_mix)]
    if len(hit):
        r = hit.sort_values('year').iloc[-1]
        L = float(r.value_pct) / 100
        rec.update(loss_year=int(r.year), L=L, loss_geo_name=r['name'],
                   loss_geo_code=iso3,
                   loss_source='invalid_L' if L >= 1 else 'country')
    else:
        if iso3 in iso_region.index:
            rcode = iso_region.loc[iso3, 'region_code']
            rname = iso_region.loc[iso3, 'region_name']
        else:
            rcode, rname = '', ''
        rhit = (region_wb[(region_wb.iso3 == rcode) & (region_wb.year <= y_mix)]
                if rcode else hit)
        if len(rhit):
            r = rhit.sort_values('year').iloc[-1]
            L = float(r.value_pct) / 100
            rec.update(loss_year=int(r.year), L=L,
                       loss_geo_name=rname or r['name'], loss_geo_code=rcode,
                       loss_source='invalid_L' if L >= 1 else 'region')
    audit_rows.append(rec)
    if rec['loss_source'] not in ('country', 'region'):
        continue
    L, y_loss = rec['L'], rec['loss_year']
    pname = at_user['ProcessName'].replace('<location>', area)
    dqi = base_dqi
    if y_loss != y_mix:
        dqi = increment_dqi_value(dqi, 2)
    if rec['loss_source'] == 'region':
        dqi = increment_dqi_value(dqi, 3)
    dqi = '(' + ';'.join(str(min(int(x), 5))
                         for x in dqi.strip('()').split(';')) + ')'
    grid_uuid = make_uuid(meta['Process']['ProcessName'].replace('<location>', area))
    common = dict(ProcessName=pname, ProcessCategory=at_user['ProcessCategory'],
                  location=loc, FlowType='PRODUCT_FLOW', unit=at_user['Unit'])
    user_rows += [
        {**common, 'FlowName': at_user['FlowName'], 'FlowUUID': at_user['FlowUUID'],
         'Context': at_user['FlowContext'], 'IsInput': False, 'reference': True,
         'amount': 1.0, 'default_provider': '', 'exchange_dqi': '',
         'description': ''},
        {**common, 'FlowName': meta['Process']['FlowName'],
         'FlowUUID': meta['Process']['FlowUUID'],
         'Context': meta['Process']['FlowContext'], 'IsInput': True,
         'reference': False, 'amount': 1 / (1 - L),
         'default_provider': grid_uuid, 'exchange_dqi': dqi, 'description': ''},
    ]
    user_info[pname] = rec

pd.DataFrame(audit_rows).to_csv(out_path / 'at_user_loss_audit.csv', index=False)
print(pd.DataFrame(audit_rows).loss_source.value_counts().to_string())

if user_rows:
    df_user = pd.DataFrame(user_rows)
    validate_exchange_data(df_user)
    lv_flow = get_single_object(
        "US Electricity Baseline", "FLOW", at_user["FlowUUID"])
    flows[lv_flow.id] = lv_flow
    if lv_flow.id not in new_flows:
        new_flows.append(lv_flow.id)
    for pname, rec in user_info.items():
        area, iso3, L = rec['Area'], rec['CountryCode'], rec['L']
        y_mix, y_loss = rec['mix_year'], rec['loss_year']
        meta_u = copy.deepcopy(user_meta_base)
        fields = dict(area=area, iso3=iso3, mix_year=y_mix, loss=L,
                      loss_year=y_loss, region_name=rec['loss_geo_name'],
                      region_code=rec['loss_geo_code'])
        for k, template in user_templates[rec['loss_source']].items():
            meta_u[k] = template.rstrip().format(**fields)
        assign_year_to_meta(meta_u, int(y_loss))
        processes.update(build_process_dict(
            df_user[df_user.ProcessName == pname],
            flows, meta=meta_u, loc_objs=location_objs,
            source_objs=source_objs, actor_objs=actor_objs, dq_objs=dq_objs))

write_objects('international_electricity', flows, new_flows, processes,
              location_objs, source_objs, actor_objs, dq_objs,
              out_path = out_path)

#%% Unzip files to repo
from flcac_utils.util import extract_latest_zip

zip_path = max(out_path.glob('international_electricity_olca2.0_*.zip'),
               key=lambda p: p.stat().st_mtime)
extract_latest_zip(zip_path,
                   parent_path,
                   output_folder_name = Path('output') / 'international_electricity_v1')
