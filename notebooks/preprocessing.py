import pandas as pd
import warnings
import numpy as np
from sklearn import set_config
set_config(transform_output="pandas")

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

### global variables ###

data_sparse_countries = ['Andorra','Antigua and Barbuda', 'Bahamas','Bosnia and Herzegovina','Brunei Darussalam','Cook Islands','Dominica','Equatorial Guinea','Faroe Islands','Grenada','Holy See','Kiribati', 'Liechtenstein','Marshall Islands','Micronesia (Federated States of)','Monaco','Montenegro','Nauru','Niue','Palau','Saint Kitts and Nevis','Saint Lucia','Saint Vincent and the Grenadines','Samoa','Serbia','San Marino', 'Sao Tome and Principe','Singapore', 'Seychelles','Solomon Islands','Tokelau','Tonga','Vanuatu','Djibouti','Maldives','Papua New Guinea','Tuvalu']

exclude_list = ['Water withdrawal for livestock (watering and cleaning)', 'Water withdrawal for aquaculture', 'Harvested irrigated temporary crop area: Sweet potatoes', 'Harvested irrigated permanent crop area: Plantains', 'Number of people undernourished (3-year average)', '% of area equipped for irrigation drained', 'Harvested irrigated temporary crop area: Millet', 'Direct use of agricultural drainage water', 'Harvested irrigated temporary crop area: Sorghum', 'Harvested irrigated temporary crop area: Cassava', 'Population affected by water related disease', 'Direct use of treated municipal wastewater for irrigation purposes', 'Harvested irrigated permanent crop area: Other fruits', 'Harvested irrigated permanent crop area: Cocoa beans', 'Harvested irrigated permanent crop area: Tea', 'Harvested irrigated permanent crop area: Rubber', 'Harvested irrigated temporary crop area: Groundnuts', 'Harvested irrigated temporary crop area: Sesame', 'Harvested irrigated permanent crop area: Olives', '% of area equipped for irrigation by direct use of agricultural drainage water ', 'Harvested irrigated permanent crop area: Coconuts', 'Area equipped for irrigation by direct use of agricultural drainage water', 'Area equipped for irrigation by desalinated water', '% of area equipped for irrigation by desalinated water']

drop_list = ['Direct use of not treated municipal wastewater for irrigation purposes', 'Not treated municipal wastewater discharged (secondary water)', 'Area equipped for irrigation by direct use of not treated municipal wastewater', 'Area equipped for irrigation by direct use of non-treated municipal wastewater ', '% of area equipped for irrigation by direct use of non-treated municipal wastewater', 'Exploitable: irregular renewable surface water', 'Harvested irrigated permanent crop area: Oil palm', 'Harvested irrigated permanent crop area: Coffee', 'Area equipped for irrigation: spate irrigation', 'Harvested irrigated temporary crop area: Flowers', 'Harvested irrigated permanent crop area: Grass and Fodder', 'Area equipped for irrigation by mixed surface water and groundwater', 'Harvested irrigated permanent crop area: Grapes', 'Flood recession cropping area non-equipped', '% of the agricultural holdings with irrigation managed by women', '% of the area equipped for irrigation managed by women', 'Area waterlogged by irrigation', 'Permanent meadows and pastures irrigated', 'Cultivated wetlands and inland valley bottoms non-equipped', 'Exploitable: regular renewable surface water', 'Harvested irrigated temporary crop area: Other roots and tubers', 'Harvested irrigated temporary crop area: Sunflower', 'Harvested irrigated temporary crop area: Soybeans', 'Area equipped for irrigation: equipped lowland areas', '% of area equipped for irrigation by direct use of  treated municipal wastewater', 'Harvested irrigated permanent crop area: Bananas', 'Harvested irrigated temporary crop area: Tobacco', 'Area equipped for irrigation by direct use of treated municipal wastewater', 'Exploitable: total renewable surface water', 'Treated municipal wastewater discharged (secondary water)', 'Harvested irrigated temporary crop area: Barley', 'Harvested irrigated temporary crop area: Sugar beet', 'Capacity of the municipal wastewater treatment facilities', 'Non-irrigated cultivated area drained', 'Harvested irrigated temporary crop area: Cotton', 'Water withdrawal for cooling of thermoelectric plants', 'Area equipped for irrigation drained', '% of total grain production irrigated', 'Collected municipal wastewater', 'Harvested irrigated temporary crop area: Other cereals', 'Harvested irrigated temporary crop area: Leguminous crops', '% of area equipped for irrigation salinized', 'Area salinized by irrigation', 'Exploitable: regular renewable groundwater', 'Harvested irrigated temporary crop area: Fodder', 'Harvested irrigated permanent crop area: Citrus', 'Harvested irrigated temporary crop area: Sugarcane', 'Not treated municipal wastewater', 'Area equipped for power irrigation (surface water or groundwater)', '% of area equipped for irrigation by mixed surface water and groundwater', 'Total exploitable water resources', '% of area equipped for irrigation power irrigated', 'Direct use of treated municipal wastewater', 'Harvested irrigated temporary crop area: Wheat', 'Seasonal variability (WRI)', 'Interannual variability (WRI)', 'Number of municipal wastewater treatment facilities', 'Flood occurrence (WRI)', 'Total cultivated area drained', '% of total cultivated area drained', 'Harvested irrigated temporary crop area: Other crops', 'Harvested irrigated temporary crop area: Potatoes', 'Desalinated water produced', 'Harvested irrigated permanent crop area: Other crops', 'Harvested irrigated temporary crop area: Rice', 'Harvested irrigated temporary crop area: Maize', 'Area equipped for full control irrigation: localized irrigation', 'Treated municipal wastewater', 'Area equipped for full control irrigation: sprinkler irrigation', 'Irrigation water withdrawal', 'Fresh surface water withdrawal', 'Produced municipal wastewater', 'Fresh groundwater withdrawal', '% of irrigation potential equipped for irrigation', 'Irrigation potential', 'Harvested irrigated permanent crop area: Total', 'Area equipped for full control irrigation: surface irrigation', 'Area equipped for irrigation by groundwater', 'Harvested irrigated temporary crop area: Vegetables', 'Irrigation water requirement', '% of area equipped for irrigation by groundwater ', 'Irrigated cropping intensity', 'Gender Inequality Index (GII) [equality = 0; inequality = 1)', 'Prevalence of undernourishment (3-year average)', 'Area equipped for irrigation by surface water', 'Harvested irrigated temporary crop area: Total', 'Total harvested irrigated crop area (full control irrigation)', 'Area equipped for irrigation: actually irrigated', '% of the area equipped for irrigation actually irrigated', '% of area equipped for irrigation by surface water']

dropdrop_list = ["GDP Deflator (2015)" , "Environmental Flow Requirements" , "Groundwater: accounted outflow to other countries" , "Groundwater: leaving the country to other countries (total)"]

initial_drop_columns = ["M49", "Symbol", "Symbol Description", "Unit"]

linear_drop_list = ["% of the cultivated area equipped for irrigation", "Area equipped for full control irrigation: actually irrigated", "Area equipped for full control irrigation: total", "MDG 7.5. Freshwater withdrawal as % of total renewable water resources", "Arable land area", "Area equipped for irrigation: total", "Long-term average annual precipitation in volume", "Long-term average annual precipitation in depth", "Groundwater produced internally","Surface water produced internally", "Permanent crops area", "Overlap between surface water and groundwater", "Overlap: between surface water and groundwater", "Rural population with access to safe drinking-water (JMP)", "Surface water: inflow not submitted to treaties", "Surface water: inflow submitted to treaties", "Surface water: accounted inflow", "Total renewable water resources", "Surface water: outflow to other countries secured through treaties", "Surface water: total external renewable", "Surface water: total flow of border rivers", "Surface water: accounted flow of border rivers", "Total renewable surface water", "% of area equipped for full control irrigation actually irrigated", "Surface water: outflow to other countries not submitted to treaties", "Surface water: outflow to other countries submitted to treaties", "Surface water: inflow secured through treaties", "Total freshwater withdrawal"]

drop_for_now_list = ["Human Development Index (HDI) [highest = 1]", "Industry, value added to GDP" , "Services, value added to GDP", "Agriculture, value added to GDP", "Industrial water withdrawal as % of total water withdrawal", "Municipal water withdrawal as % of total withdrawal", "Gross Domestic Product (GDP)", "Urban population with access to safe drinking-water (JMP)"]

complete_drop_list = exclude_list + drop_list + dropdrop_list + linear_drop_list + drop_for_now_list

transformation_dict = {
    "log":
        ["agricultural_water_withdrawal","agricultural_water_withdrawal_as_%_of_total_renewable_water_resources",        "cultivated_area_arable_land_plus_permanent_crops", "dam_capacity_per_capita", "groundwater_entering_the_country_total", "industrial_water_withdrawal", "municipal_water_withdrawal", "population_density", "ratio_between_rainfed_and_irrigated_yields", "rural_population", "industrial_water_use_efficiency","irrigated_agriculture_water_use_efficiency", "services_water_use_efficiency","water_use_efficiency","surface_water_entering_the_country_total", "surface_water_leaving_the_country_to_other_countries_total", "total_agricultural_water_managed_area","total_area_of_the_country_excl_coastal_water", "total_dam_capacity", "total_internal_renewable_water_resources", "total_internal_renewable_water_resources_per_capita", "total_population", "total_renewable_groundwater", "total_renewable_water_resources_per_capita", "total_water_withdrawal", "urban_population"],
    "square_root":
        ["national_rainfall_index"],
    "cube_root":
        ["%_of_agricultural_gva_produced_by_irrigated_agriculture", "%_of_total_country_area_cultivated",  "agriculture_value_added_%_gdp", "total_water_withdrawal_per_capita"]}

imputation_dict = {
    'Qatar': {'dam_capacity_per_capita': 0, 'total_dam_capacity': 0,'ratio_between_rainfed_and_irrigated_yields': 0},
    'Republic of Korea': {'national_rainfall_index': 1150},
    'Saudi Arabia': {'ratio_between_rainfed_and_irrigated_yields': 0, 'surface_water_leaving_the_country_to_other_countries_total': 0},
    'South Sudan': {'national_rainfall_index': 0, 'total_dam_capacity': 0, 'dam_capacity_per_capita': 0},
    'Sudan': {'national_rainfall_index': 257},
    'Timor-Leste':{'national_rainfall_index': 1500, 'total_dam_capacity': 0, 'dam_capacity_per_capita': 0},
    'Turkmenistan': {'ratio_between_rainfed_and_irrigated_yields': 0},
    'United Arab Emirates': {'ratio_between_rainfed_and_irrigated_yields': 0},
    'Yemen': {'surface_water_leaving_the_country_to_other_countries_total': 0},
    'Iceland': {'%_of_agricultural_water_managed_area_equipped_for_irrigation': 0, 'total_agricultural_water_managed_area': 0},
    'Iraq': {'surface_water_leaving_the_country_to_other_countries_total': 0},
    'Israel': {'dam_capacity_per_capita': 0,'total_dam_capacity': 0},
    'Jordan': {'surface_water_leaving_the_country_to_other_countries_total': 0},
    'Kuwait': {'dam_capacity_per_capita': 0, 'total_dam_capacity': 0, 'ratio_between_rainfed_and_irrigated_yields': 1.24},
    'Malta': {'national_rainfall_index': 568},
    'Luxembourg': {'national_rainfall_index': 847.9},
    'Mauritius': {'national_rainfall_index': 2342},
    'Oman': {'ratio_between_rainfed_and_irrigated_yields': 1.24,'surface_water_leaving_the_country_to_other_countries_total': 0},
    'Palestine': {'national_rainfall_index': 490},
    'Azerbaijan': {'surface_water_leaving_the_country_to_other_countries_total': 0},
    'Bahrain': {'national_rainfall_index': 36},
    'Barbados': {'dam_capacity_per_capita': 0, 'total_dam_capacity': 0, 'national_rainfall_index': 1525},
    'Burundi': {'dam_capacity_per_capita': 0, 'total_dam_capacity': 0},
    'Central African Republic': {'dam_capacity_per_capita': 0, 'total_dam_capacity': 0},
    'Chad': {'dam_capacity_per_capita': 0, 'total_dam_capacity': 0},
    'Comoros': {'dam_capacity_per_capita': 0, 'total_dam_capacity': 0},
    "Democratic People's Republic of Korea": {'national_rainfall_index': 244},
    'Egypt': {'ratio_between_rainfed_and_irrigated_yields': 1.124},
    'Gambia': {'dam_capacity_per_capita': 0, 'total_dam_capacity': 0},
    'Puerto Rico': {'dam_capacity_per_capita': 119.6, 'total_dam_capacity': 0.3937}
    }

scaled_cols = ['year', '%_of_agricultural_gva_produced_by_irrigated_agriculture', '%_of_agricultural_water_managed_area_equipped_for_irrigation', '%_of_total_country_area_cultivated', 'agricultural_water_withdrawal', 'agricultural_water_withdrawal_as_%_of_total_renewable_water_resources', 'agricultural_water_withdrawal_as_%_of_total_water_withdrawal', 'agriculture_value_added_%_gdp', 'cultivated_area_arable_land_plus_permanent_crops', 'dam_capacity_per_capita', 'dependency_ratio', 'groundwater_accounted_inflow', 'groundwater_entering_the_country_total', 'industrial_water_withdrawal', 'municipal_water_withdrawal', 'national_rainfall_index', 'population_density', 'ratio_between_rainfed_and_irrigated_yields', 'rural_population', 'industrial_water_use_efficiency', 'irrigated_agriculture_water_use_efficiency', 'services_water_use_efficiency', 'water_use_efficiency', 'surface_water_entering_the_country_total', 'surface_water_leaving_the_country_to_other_countries_total', 'total_agricultural_water_managed_area', 'total_area_of_the_country_excl_coastal_water', 'total_dam_capacity', 'total_internal_renewable_water_resources', 'total_internal_renewable_water_resources_per_capita', 'total_population', 'total_renewable_groundwater', 'total_renewable_water_resources_per_capita', 'total_water_withdrawal', 'total_water_withdrawal_per_capita', 'urban_population', 'water_resources_total_external_renewable']

hot_cols = ['country']


### drop class ###

class ColumnSlicer:
    def __init__(self, new_slice_list):
        self.new_slice_list = new_slice_list
    def fit(self, df, y=None):
        pass
    def transform(self, df, y=None):
        [[r0, r1], [c0, c1]] = self.new_slice_list 
        df_new = df.iloc[r0:r1, c0:c1]
        return df_new
    def fit_transform(self, df, y=None):
        self.fit(df)
        return self.transform(df)

class ColumnDropper:
    def __init__(self, to_drop_columns):
        self.to_drop_columns = to_drop_columns
    def fit(self, df, y=None):
        pass
    def transform(self, df, y=None):
        df_new = df.drop(columns=self.to_drop_columns)
        return df_new
    def fit_transform(self, df, y=None):
        self.fit(df)
        return self.transform(df)

class FromColumnDropper:
    def __init__(self, drop_from_col, drop_list):
        self.drop_from_col = drop_from_col
        self.drop_list = drop_list

    def fit(self, df, y=None):
        pass

    def transform(self, df, y=None):
        #print("df", df.shape)
        df_short = df[~df[self.drop_from_col].isin(self.drop_list)]
        #print("df_short", df_short.shape)
        return df_short

    def fit_transform(self, df, y=None):
        self.fit(df)
        return self.transform(df)


### pivot class ###

class Pivoter:
    def __init__(self):
        pass

    def fit(self, df, y=None):
        pass

    def transform(self, df, y=None):
        df_pivot = df.pivot_table(index=["Country","Year"], columns="Variable", values="Value")
        df_pivot.reset_index(inplace=True)
        return df_pivot

    def fit_transform(self, df, y=None):
        self.fit(df)
        return self.transform(df)


### imputer classes ###

class GroupedMedianImputer:
    def __init__(self, group_col, value_cols):
        self.group_col = group_col
        self.value_cols = value_cols
        self.medians = None

    def fit(self, df, y=None):
        self.medians = df.groupby(self.group_col)[self.value_cols].median()

    def transform(self, df, y=None):
        df_new = df.copy()
        if self.medians is None:
            raise Exception("The imputer has not been fitted yet.")
        
        def fillna_grouped(x):
            return x.fillna(self.medians.loc[x.name, col])

        for col in self.value_cols:
            if col==self.group_col: continue
            df_new[col] = df.groupby(self.group_col)[col].transform(fillna_grouped)
        
        return df_new

    def fit_transform(self, df, y=None):
        self.fit(df)
        return self.transform(df)


class FromDictImputer:
    def __init__(self, imputation_dict):
        self.imputation_dict = imputation_dict

    def fit(self, df, y=None):
        pass

    def transform(self, df, y=None):
        df_new = df.copy()
        for c, c_dict in self.imputation_dict.items():
            #print(c)
            if c in df_new.country.to_list():
                index_list = df_new.query("country == @c").index
                for col, val in c_dict.items():
                    #print(df_new.loc[index_list, col])
                    df_new.loc[index_list, col] = val
                    #print(df_new.loc[index_list, col])

        return df_new

    def fit_transform(self, df, y=None):
        self.fit(df)
        return self.transform(df)


### converter classes / functions ###

class FromDictConverter:
    def __init__(self, conversion_dict):
        self.conversion_dict = conversion_dict

    def fit(self, df, y=None):
        pass
    
    def transform(self, df, y=None):
        df_new = df.copy()
        df_new = self.log_convert(df_new, self.conversion_dict["log"])
        df_new = self.sqroot_convert(df_new, self.conversion_dict["square_root"])
        df_new = self.cube_convert(df_new, self.conversion_dict["cube_root"])
        return df_new

    def log_convert(self, df, col_list):
        df[col_list] = df[col_list].applymap(lambda x: np.log(x+1e-10))
        return df

    def sqroot_convert(self, df, col_list):
        df[col_list] = df[col_list].applymap(np.sqrt)
        return df

    def cube_convert(self, df, col_list):
        df[col_list] = df[col_list].applymap(lambda x: np.power(x, 1/3.))
        return df

    def fit_transform(self, df, y=None):
        self.fit(df)
        return self.transform(df)

### renaming classes ###

class ColumnRenamer:
    def __init__(self):
        pass
    def fit(self, df, y=None):
        pass
    def transform(self, df, y=None):
        df_new = df.copy()

        colnames = []
        for col in df_new.columns:
            col = col.replace(" ","_")
            col = col.replace("_(NRI)","")
            col = col.replace("_(GDP)","")
            col = col.replace("_(JMP)","")
            col = col.replace("_(IRWR)","")
            col = col.replace("SDG_6.4.1._","")
            col = col.replace("SDG_6.4.2._","")
            col = col.replace("-","_")
            col = col.replace(":","")
            col = col.replace(".","")
            col = col.replace(",","")
            col = col.replace("(","")
            col = col.replace(")","")
            col = col.lower()
            col = col.replace("+","plus")
            colnames.append(col)

        df_new.columns = colnames

        return df_new

    def fit_transform(self, df, y=None):
        self.fit(df)
        return self.transform(df)

class ColTransNameRemover:
    def __init__(self):
        pass
    def fit(self, df, y=None):
        pass
    def transform(self, df, y=None):
        df_new = df.copy()
        colnames = []
        for col in  df_new.columns.to_list():
            col = col.split("__")[-1]
            colnames.append(col)
        df_new.columns = colnames
        return df_new
    def fit_transform(self, df, y=None):
        self.fit(df)
        return self.transform(df)


### preprocessing getter functions ###

def get_pre_split_processor():

    pre_split_pipeline = Pipeline([
        ("initial_drop_columns", ColumnDropper(initial_drop_columns)),
        ("drop_countries", FromColumnDropper("Country", data_sparse_countries)),
        ("drop_variables", FromColumnDropper("Variable", complete_drop_list)),
        ("pivot", Pivoter()),
        ("rename", ColumnRenamer()),
    ])

    return pre_split_pipeline

def get_full_preprocessor(full_column_list, hot=True):

    hot_columns = ColumnTransformer([
        ('num', StandardScaler(), scaled_cols),
        ('cat', OneHotEncoder(drop='first', sparse_output=False), hot_cols),
        ])
    scale_columns = ColumnTransformer([
        ('num', StandardScaler(), scaled_cols),
        ], remainder="passthrough")

    if hot:
        preprocessor = Pipeline([
            ("hard_imputation", FromDictImputer(imputation_dict)),
            ("median_imputation", GroupedMedianImputer("country", [col for col in full_column_list if not col=="country"])),
            ("conversion", FromDictConverter(transformation_dict)),     
            ("column_transformation", hot_columns)   
        ])
    else:
        preprocessor = Pipeline([
            ("hard_imputation", FromDictImputer(imputation_dict)),
            ("median_imputation", GroupedMedianImputer("country", [col for col in full_column_list if not col=="country"])),
            ("conversion", FromDictConverter(transformation_dict)),     
            ("column_transformation", scale_columns),
            ("remove_col_trans_names",  ColTransNameRemover())
        ])

    return preprocessor
