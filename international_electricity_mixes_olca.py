"""
Generate processes of international electricity mixes
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
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
      .query('`Area type` == "Country"')
      .query('Category == "Electricity generation"')
      .query('Subcategory == "Fuel"')
      .query('Unit == "TWh"')
      .filter(['Year', 'Area', 'Country', 'Country code',
               'Variable', 'Unit', 'Value'])
      .rename(columns={'Country code': 'CountryCode',
                       'Variable': 'Fuel'})
      )
df['share'] = df['Value'] / df.groupby(['CountryCode', 'Year'])['Value'].transform('sum')

countries = (df[['Area', 'CountryCode', 'Year']]
             .sort_values(by='Year')
             .drop_duplicates(subset=['Area', 'CountryCode'], keep='last')
             )
# countries.to_csv(parent_path / 'country_corr.csv', index=False)

# merge back in to keep only the latest set of data for each area
df = df.merge(countries, how='inner')
df = df.dropna(subset='share')

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
           .query('CountryCode != "USA"') # Drop US data?
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
    process_meta = yaml.safe_load(f)

(process_meta, source_objs) = extract_sources_from_process_meta(
    process_meta, bib_path = data_path / 'electricity.bib')
(process_meta, actor_objs) = extract_actors_from_process_meta(process_meta)
dq_objs = extract_dqsystems(meta['DQI']['dqSystem'])
process_meta['dq_entry'] = format_dqi_score(meta['DQI']['Process'])

# generate dictionary of location objects
location_objs = build_location_dict(df_olca, locations)

#%% Create json file
from flcac_utils.generate_processes import build_flow_dict, \
    build_process_dict, write_objects, validate_exchange_data
from flcac_utils.util import assign_year_to_meta

validate_exchange_data(df_olca)
flows, new_flows = build_flow_dict(df_olca)
processes = {}
for year in df_olca.Year.unique():
    process_meta = assign_year_to_meta(process_meta, year)
    # Update time period to match year for each region

    p_dict = build_process_dict(df_olca.query('Year == @year'),
                                flows,
                                meta=process_meta,
                                loc_objs=location_objs,
                                source_objs=source_objs,
                                actor_objs=actor_objs,
                                dq_objs=dq_objs,
                                )
    processes.update(p_dict)

write_objects('international_electricity', flows, new_flows, processes,
              location_objs, source_objs, actor_objs, dq_objs,
              out_path = out_path)
