"""
Functions for Calculator
"""

import streamlit as st
from py50.calculator import Calculator

class Calc_Logic:
    def __init__(self):
        pass

    # todo create button in the utils script
    # Set up download button for csv files
    def download_button(self, df, file_name=None):
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button('Download table as CSV', data=csv, file_name=file_name, mime='text/csv')


    def calculator_program(self, df=None, paste=True):
        '''
        Program to run calculations. It must include nested if/else depending
        on whether the input dataframe is pasted or a csv file.

        :param df: Input dataframe
        :return: editable dataframe
        '''
        st.sidebar.markdown(':green[**Calculator Options:**]')

        # Logic based on paste or CSV input
        if paste is True:
            # Select columns for calculation
            drug_query = df
            col_header = drug_query.columns.to_list()
            drug_name = st.selectbox('Drug Name:', (col_header))
            compound_conc = st.selectbox('Drug Concentration:', (col_header), index=1)  # Index to auto select column
            ave_response = st.selectbox('Average Response column:', (col_header), index=2)  # Index to auto select column
        else:
            drug_query = df
            col_header = drug_query.columns.to_list()
            drug_name = st.selectbox('Drug Name:', (col_header))
            compound_conc = st.selectbox('Drug Concentration:', (col_header))
            ave_response = st.selectbox('Average Response column:', (col_header))

        # sidebar options
        units = st.sidebar.radio('Input Concentration Units',
                                 options=['nM', 'µM'],
                                 captions=['Nanomolor', 'Micromolar'])

        # Set conditions for calculations
        conditions = {drug_name, compound_conc, ave_response}
        if len(conditions) != 3:
            st.write('### :red[Select Drug, Concentration, and Response Columns!]')
        else:
            st.write('## Filtered Table')
            df_calc = drug_query.filter(items=(drug_name, compound_conc, ave_response), axis=1)
            # Ensure columns are float
            df_calc[compound_conc] = df_calc[compound_conc].astype(float)
            df_calc[ave_response] = df_calc[ave_response].astype(float)

            # Output filtered table for calculation
            st.data_editor(df_calc, num_rows='dynamic')

            # Calculate IC50
            data = Calculator(df_calc)

            absolute = data.calculate_absolute_ic50(name_col=drug_name,
                                                    concentration_col=compound_conc,
                                                    response_col=ave_response,
                                                    input_units=units)
            st.markdown('## Calculated Results')

            # Absolute IC50 Table
            st.data_editor(absolute, num_rows='dynamic')
            self.download_button(absolute, file_name='py50_ic50.csv')

            # output unit message
            st.markdown(f'Calculated Results are in {units}')

            convert = st.checkbox('Calculate values to pIC50?')

            if convert is True:
                st.markdown('## Calculated Results with pIC50')
                conversion = data.calculate_pic50(name_col=drug_name,
                                                  concentration_col=compound_conc,
                                                  response_col=ave_response,
                                                  input_units=units)
                # Output pIC50 table
                st.data_editor(conversion, num_rows='dynamic')
                self.download_button(conversion, file_name='py50_pic50.csv')