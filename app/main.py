from utils import *
def main():

    # Them  config streamlit
    streamlit.set_page_config(layout='wide')
    set_page_container_style()
    # read the configuration file and initialize random generators
    cfg = read_config('/app/mib_kurs_analyse/app/config/config.yaml')

    # read dataframes
    df_termine, df_studenten, df_studentenxtermine, df_pk_kurs = generate_dataframes(cfg)

    # Kurs Info
    # Kurs dataframe
    all_dates_kurse = df_termine[cfg['plots']['kurse']['fields']]

    # Studenten ID
    kurs_id = str(get_param('kurs_id'))

    # Select value range for courses
    course_dates = all_dates_kurse[all_dates_kurse['MiB-Kurs-Name'] == kurs_id]
    course_dates = course_dates['Datum-df']
    min_date_kurs, max_date_kurs = min_max_dates(course_dates)

    col_date_1, col_date_2 = streamlit.columns(2)
    with col_date_1:
        start_date = streamlit.date_input('Anfangsdatum', min_value=min_date_kurs,
                                        max_value=max_date_kurs, value=min_date_kurs)
    with col_date_2:
        end_date = streamlit.date_input('Enddatum', min_value=min_date_kurs,
                                        max_value=max_date_kurs, value=max_date_kurs)

    plot_kurs_data(df_termine, df_pk_kurs, cfg, kurs_id, start_date, end_date)


if __name__ == "__main__":
    main()
