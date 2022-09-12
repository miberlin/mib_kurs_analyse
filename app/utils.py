import time
import numpy
import numpy as np
import requests
import pandas
from ruamel.yaml import YAML
import matplotlib.pyplot
import streamlit
from scipy.interpolate import interp1d
from datetime import datetime

AIRTABLE_API_KEY = streamlit.secrets['AIRTABLE_API_KEY']


# @streamlit.cache
def read_config(filename):
    with open(filename) as f:
        yaml = YAML(typ='safe')
        cfg = yaml.load(f)
    return cfg


def list_airtable_records(base, table, cfg):
    # Get info from config
    api_url = cfg['api_url']
    base_id = cfg[base]['id']
    table_dict = cfg[base][table]
    table_id = table_dict['id']
    table_name = table_dict['name']
    fields = table_dict['fields']
    #Composing URLs
    table_url = f'{api_url}{base_id}/{table_id}'
    endpoint = f'{table_url}?view=Python'  #maxRecords=100000&view=Python'
    # python requests headers
    headers_list = {
        "Authorization": f'Bearer {AIRTABLE_API_KEY}'
    }
    offset = '0'
    counter = 0
    result = []
    while True:
        querystring = {
            "offset": offset}
        counter += 1
        try:
            response = requests.get(endpoint, params=querystring, headers=headers_list)
            response_table = response.json()
            records = response_table['records']
            result = result + records
            if counter > 5:
                time.sleep(1)
                counter = 0

            try:
                offset = response_table['offset']
            except Exception:
                break
        except Exception as e:
            print(counter, e)

    df1 = pandas.DataFrame.from_dict(result)
    df1 = df1.set_index('id')
    df = pandas.concat([df1.drop(['fields'], axis=1), df1['fields'].apply(pandas.Series)], axis=1)
    if table_name == "Studenten Allgemein":
        df = df.set_index('MiB-ID')
        # Array to string
        # which_array_to_str = ["MiB-Kurse", "Uni-Module", "Studenten x Termine"]
        # for cat in which_array_to_str:
        #     df[cat] = df[cat].str[0]
    elif table_name == "Studenten":
        df = df.set_index('MiB-ID')
        for cat in table_dict['to_clean']:
            df[cat] = df[cat].str[0]
    elif table_name == "Studenten x Termine":
        for cat in table_dict['to_clean']:
            df[cat] = df[cat].str[0]
        # Fill nan with False
        for cat in table_dict['fill_nan']:
            df[cat] = df[cat].fillna(value=False)
    elif table_name == "Termine":
        # Array to string
        for cat in table_dict['to_clean']:
            df[cat] = df[cat].str[0]
        for cat in table_dict['clean_nan']:
            df[cat] = df[cat]#.astype('float64')
    elif table_name == "PK Ergebnisse":
        for cat in table_dict['to_clean']:
            df[cat] = df[cat].str[0]

    df = df[fields]  # Reduce db
    return df


# @streamlit.cache
def generate_dataframes(cfg):
    df_studenten = list_airtable_records('test_ws2122_base', 'studenten_table', cfg)
    df_studentenxtermine = list_airtable_records('test_ws2122_base', 'studentenxtermine_table', cfg)
    # df_mibkurse = list_airtable_records('test_ws2122_base', 'mibkurse_table', cfg)
    df_termine = list_airtable_records('test_ws2122_base', 'termine_table', cfg)
    df_pk_stud = list_airtable_records('test_ws2122_base', 'pkergebnisse_table', cfg)
    return df_termine, df_studenten, df_studentenxtermine, df_pk_stud

# headers_create = {
#         "Authorization": "Bearer key2rtabfsnUeI5pi",
#         "Content-Type": "application/json"
#     }
# http methods
# GET: The GET method requests a representation of the specified resource. Requests using GET should only retrieve data.
# HEAD: The HEAD method asks for a response identical to that of a GET request, but without the response body.
# POST: The POST method is used to submit an entity to the specified resource, often causing a change in state or
#       side effects on the server.

# df = pandas.DataFrame(r.json())
# print(df)

# Data visualization

# Student plots


# @streamlit.cache # Interpolations
def interpolation_values(number_of_values, series):
    x = np.linspace(0, number_of_values - 1, num=len(series.dropna()))
    y = series.dropna()
    f = interp1d(x, y, kind='linear')
    xnew = numpy.linspace(0, number_of_values - 1, 50)
    ynew = f(xnew)
    return xnew, ynew


# @streamlit.cache
def min_max_dates(df):
    dates = df.unique()
    dates = pandas.to_datetime(dates, format='%d/%m/%y').sort_values()
    dates = dates.strftime('%d/%m/%y')
    min_date = dates[0]
    max_date = dates[-1]
    min_date = datetime.strptime(min_date, '%d/%m/%y')
    max_date = datetime.strptime(max_date, '%d/%m/%y')
    return min_date , max_date


# @streamlit.cache
def missing_values_plot(df):
    fill1 = df.ffill()
    fill1.fillna(method='bfill')
    fill2 = df.bfill()
    fill2.fillna(method='ffill')
    df = (fill1 + fill2) / 2
    return df


# @streamlit.cache
def kurs_plot_data_options(df, df_pk,cfg,kurs_id,start_date,end_date):
    # Student data
    kurs_data = df[cfg['plots']['kurse']['fields']]
    kurs_data['Art des Termins'] = kurs_data['Art des Termins'].fillna('PK')
    values_name = kurs_data[kurs_data['MiB-Kurs-Name'] == kurs_id]
    values_name['Datum-df'] = pandas.to_datetime(values_name['Datum-df'], format='%d/%m/%y')
    values_name.sort_values(by='Datum-df', inplace=True)
    values_name['Datum-df'] = values_name['Datum-df'].dt.date
    values_name['Datum-df'] = values_name['Datum-df'].loc[(start_date <= values_name['Datum-df']) &
                                                          (end_date >= values_name['Datum-df'])]
    values_name['Datum-df'] = pandas.to_datetime(values_name['Datum-df'])
    values_name['Datum-df'] = values_name['Datum-df'].dt.strftime('%d/%m/%y')
    dates = values_name['Datum-df']

    values_name = values_name.loc[values_name['Datum-df'].isin(dates)]

    number_of_values = values_name.shape[0]
    values_range = numpy.linspace(0, number_of_values - 1, num=number_of_values)
    anw = values_name['Anwesenheit (from Studenten x Termine)']
    total_students = values_name['Studentenanzahl (from Studenten x Termine)']
    anw_percent = anw / total_students
    aufm = numpy.array(values_name['Aufmerksamkeit Mittel Prozentual'])
    vers = numpy.array(values_name['Verständnis Mittel Prozentual'])
    fun = nnumpy.array(values_name['Fun Mittel Prozentual'])

    pk_arr = numpy.array(values_name[['Art des Termins', 'Datum-df']])
    pk_idx = numpy.argwhere(pk_arr[:, 0] == 'PK').flatten()
    pk_datum = pk_arr[:, 1][pk_idx]
    pk_data = pandas.DataFrame(data={'PK-Index': pk_idx, 'Datum-df': pk_datum}, index=numpy.arange(len(pk_idx)))

    pk_result_data = df_pk[cfg['plots']['students']['pk_results']]
    pk_result_data = pk_result_data[pk_result_data['MiB-Kurs-Name'] == kurs_id]
    pk_result_data['Datum-df'] = pandas.to_datetime(pk_result_data['Datum-df'], format='%d/%m/%y')
    pk_result_data.sort_values(by='Datum-df', inplace=True)
    pk_result_data['Datum-df'] = pk_result_data['Datum-df'].dt.date
    pk_result_data['Datum-df'] = pk_result_data['Datum-df'].loc[(start_date <= pk_result_data['Datum-df']) &
                                                          (end_date >= pk_result_data['Datum-df'])]
    pk_result_data['Datum-df'] = pandas.to_datetime(pk_result_data['Datum-df'])
    pk_result_data['Datum-df'] = pk_result_data['Datum-df'].dt.strftime('%d/%m/%y')
    pk_result_data['PK-Index']=np.nan

    for idx in range(pk_data.shape[0]):
        datum = pk_data['Datum-df'][idx]
        index = pk_data['PK-Index'][idx]
        rows = pk_result_data.index[pk_result_data['Datum-df']==datum].tolist()
        pk_result_data['PK-Index'][rows]=index
    pk_result_data['Erreicht Prozentual'] = pk_result_data['Erreicht Prozentual']*100
    pk_data = pk_result_data.reset_index()
    pk_data['PK-Index']= pk_data['PK-Index'].astype('int')
    pk_data = pk_data[['Erreicht Prozentual','PK-Index']]
    return values_range, anw_percent, aufm, vers, fun, dates, pk_data


def plot_kurs_data(df, df_pk, cfg, kurs_id,start_date,end_date):

    # define columns in dashboard
    col1, col2, col3 = streamlit.columns(3)

    # Line selection using checkboxes
    with col1:
        aufm_checkbox = streamlit.checkbox('Aufmerksamkeit', value=True)
        vers_checkbox = streamlit.checkbox('Verständnis', value=True)
    with col2:
        fun_checkbox = streamlit.checkbox('Fun Faktor', value=True)
        anw_checkbox = streamlit.checkbox('Anwesenheit', value=True)
    with col3:
        pk_checkbox = streamlit.checkbox('PK Ergebnisse', value=True)

    fig,ax = matplotlib.pyplot.subplots()
    fig.set_size_inches(16,8)
    # plot options
    values_range, anw_percent, aufm, vers, fun, dates, pk_data = kurs_plot_data_options(df,df_pk,cfg,kurs_id,
                                                                                        start_date,end_date)
    if aufm_checkbox:
        ax.plot(values_range,100*aufm,label='Aufmerksamkeit',linewidth=4,
                linestyle='-',color='orange',zorder=-1)
    if vers_checkbox:
        ax.plot(values_range,100*vers,label='Verständnis',linewidth=4,
                linestyle='-',color='green',zorder=-1)
    if fun_checkbox:
        ax.plot(values_range,100*fun,label='Fun',linewidth=4,
                linestyle='-',color='blue',zorder=-1)
    if anw_checkbox:
        ax.plot(values_range,100*anw_percent,label='Anwesenheit',linewidth=4,
                linestyle='--',color='black')
    if pk_checkbox:
        test = pk_data.groupby('PK-Index', sort=False)['Erreicht Prozentual'].apply(list)
        pk_idx = pk_data['PK-Index'].unique()
        test = np.array(test)
        ax.violinplot(dataset = test, positions=pk_idx)

    matplotlib.pyplot.legend(fontsize=14)
    matplotlib.pyplot.xticks(values_range,labels=dates,fontsize=12,rotation=45)
    matplotlib.pyplot.yticks(numpy.linspace(0,100,5),labels = ['0 %', '25 %', '50 %', '75 %', '100 %'],fontsize=12)
    matplotlib.pyplot.grid(linewidth=.4)

    # Show figure in Dashboard
    streamlit.pyplot(fig)


# Used for the variables wich are going to be prefill using JS
def get_param(param_name):
    query_params = streamlit.experimental_get_query_params()
    try:
        return query_params[param_name][0]
    except:
        streamlit.write('Parameters is missing')
        return False


def get_params(params_names_list):
    query_params = streamlit.experimental_get_query_params()
    responses = []
    for parameter in params_names_list:
        try:
            responses.append(query_params[parameter][0])
        except Exception as e:
            responses.append(None)
    return responses

# Page style

def set_page_container_style():
    streamlit.markdown(
        f'''
                    <style>
                        .block-container {{
                            padding-top: 0rem !important;
                        }}
                        .stApp {{
                            font-family: 'Montserrat';
                            color: black !important;
                            background-color: white !important;
                        }}
                        .css-14xtw13 {{
                            display: none !important;
                        }}
                        .css-18ni7ap {{
                            display: none !important;
                        }}
                    </style>
                    ''',
        unsafe_allow_html=True,
    )