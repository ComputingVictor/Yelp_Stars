import json
import pandas as pd
from glob import glob 
import ast
import numpy as np
from pandas import Timedelta


def convert(x):
    
    """   
    Convert a JSON string containing nested lists and dictionaries into a dictionary
    with simpler keys and simple values.
    :param x: JSON file.
    :return: Modified dictionary.
    
    """ 
    
    ob = json.loads(x)
    for k, v in ob.copy().items():
        if isinstance(v, list):
            ob[k] = ','.join(v)
        elif isinstance(v, dict):
            for kk, vv in v.items():
                ob['%s_%s' % (k, kk)] = vv
            del ob[k]
    return ob



def json_to_parquet(path):
    
    """   
    Converts a series of JSON files to Parquet files using the Pandas library and
    the "convert" function defined above to process the data.
    :param path: Path with the JSON data to convert.
    :return: .parquet files.
    
    """ 
    

    for json_filename in glob(path):
        parquet_filename = '%s.parquet' % json_filename[:-5]
        print('Converting %s to %s' % (json_filename, parquet_filename))
        df = pd.DataFrame([convert(line) for line in open(json_filename, encoding='utf-8')])
        df.to_parquet(parquet_filename, index=False)
        
        
def unique_values(df, column):
    
    """ 
    Returns a list with the distinct values of the variable in the dataframe.
    :param df: Dataset to analyze
    :param column: Objective variable
    :return: List
    
    """ 
    unique_values = df[column].unique().tolist()
    return unique_values


def replace_values(df):
    """
    Replace the repeated values of the categorical variables for a common one reducing the unique values.
    :param df: DataFrame. Dataframe where the replace will be done.
    :return: DataFrame.
    
    """
    for column in df:
        
        df[column] = df[column].replace(["u'full_bar'","'full_bar'"], 'full_bar')
        df[column] = df[column].replace(["u'beer_and_wine'","'beer_and_wine'"], 'beer_and_wine')
        df[column] = df[column].replace(["u'average'","'average'"], 'average')
        df[column] = df[column].replace(["u'quiet'","'quiet'"], 'quiet')
        df[column] = df[column].replace(["u'loud'","'loud'"], 'loud')
        df[column] = df[column].replace(["u'very_loud'","'very_loud'"], 'very_loud')
        df[column] = df[column].replace(["u'casual'","'casual'"], 'casual')
        df[column] = df[column].replace(["u'formal'","'formal'"], 'formal')
        df[column] = df[column].replace(["u'dressy'","'dressy'"], 'dressy')
        df[column] = df[column].replace(["u'outdoor'","'outdoor'"], 'outdoor')
        df[column] = df[column].replace(["u'no'","'no'"], 'no')
        df[column] = df[column].replace(["u'yes'","'yes'"], 'yes')
        df[column] = df[column].replace(["u'yes_free'","'yes_free'"], 'yes_free')
        df[column] = df[column].replace(["u'yes_corkage'","'yes_corkage'"], 'yes_corkage')
        df[column] = df[column].replace("u'21plus'", '21_plus')
        df[column] = df[column].replace("u'allages'", 'all_ages')
        df[column] = df[column].replace("u'18plus'", '18_plus')
        df[column] = df[column].replace(["u'free'","'free'"], 'free')
        df[column] = df[column].replace(["u'paid'","'paid'"], 'paid')
               
        
    return df

def replace_nan_values(df):
        
        """
        Replace the NaN values of the categorical variables for a common one reducing the unique values.
        :param df: DataFrame. Dataframe where the replace will be done.
        :return: DataFrame with values NaN converted to 0.
        
        """
        for column in df:
            
            df[column] = df[column].replace(["NaN"], 0)
            
        return df

def parser(line):
    
    """ 
    Parse the categorical variables of 'attributes' columns.
    
    :param line: str. String to parse.
    :return: dict. Dictionary with the parsed values.
    
     """
    if isinstance(line, str) and line != 'None':
        line = ast.literal_eval(line)
    else:
        line = {'amb_empty': 1}

    return line


def modify_column_titles(df, prefix):
    
    """
    Modify the column titles of a dataframe by adding a prefix to each title.

    :param df: DataFrame. Dataframe whose column titles will be modified.
    :param prefix: str. Prefix to add to each column title.
    :return: DataFrame. Dataframe with modified column titles.
    """

    col_titles = list(df.columns)
    modified_col_titles = [prefix + title for title in col_titles]
    df.columns = modified_col_titles
    
    return df


def join_skip_first(list_df):
    
    """
    Join all the columns of a list of dataframes except the first one.
    :param list_df: list of dataframes
    :return: dataframe
    """
    df_resultado = pd.DataFrame()

    for df in list_df:
        
        columnas_a_agregar = df.columns[1:] 
        df_resultado[columnas_a_agregar] = df[columnas_a_agregar]
  

    return df_resultado



def replace_binary_values(df):
    
    """
    Replace 'None', 'True' and 'False' strings for None, 1 and 0 respectively in the specified columns of a dataframe.

    :param df: DataFrame. Dataframe in which the replacement will be made.
    :return: DataFrame. Dataframe with the 'None', 'True' and 'False' strings replaced by None, 1 and 0 respectively.
    
    """
    for column in df:
        df[column] = df[column].replace(['None',"'none'", "u'none'",  None], np.nan)
        df[column] = df[column].replace(['True', True], int(1))
        df[column] = df[column].replace(['False',False], int(0))
    return df


def interval(x):
    
    """

    Calculate the interval between two dates.

    :param: x: Series. Series with the start and end dates.
    :return: Timedelta. Interval between the start and end dates.

    """

    if x['end'] < x['start']:
        x['end'] += Timedelta(days=1)
    return x['end'] - x['start']


def extract_total_hours(final_df, values):

    """
    
    This function extract the total hours of the business.

    :param final_df: Dataframe with the business information.
    :param values: List with the variables that include the day hours of the week.
    :return: Dataframe with the total hours of the business joined to the original table.
    
    """
    
    
    for day in values:
        
        df = pd.DataFrame()
        df[['start', 'end']] = final_df[day].str.split('-', expand=True)

        # Create a DF with format hour.

        df['start'] = pd.to_datetime(df['start'],format='%H:%M')
        df['end'] = pd.to_datetime(df['end'],format='%H:%M')

        # Calculate the difference:
        title = str('total_') + day
        df[title] = df.apply(interval, axis=1)
        df[title] = (df[title].dt.total_seconds() / 3600).round(2)

        # Add the categories with the rest of variables.

        final_df = pd.concat([final_df, df[title]], axis=1)
    
    return final_df


def count_checkins(x):
    """
    This function counts the number of checkins per business.
    :param x: string with the checkins.
    :return: number of checkins.
    """
    values = x.split(',')
    return len(values)

