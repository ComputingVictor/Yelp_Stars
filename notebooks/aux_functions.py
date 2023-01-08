
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import warnings
from scipy.stats import chi2_contingency
import pickle
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix, roc_curve
import sklearn




def f2_func(y_true, y_pred):
    
    """   
    Obtain the F2 parameter between true and predict value.
    :param ytrue: List with real values.
    :param ypred: List with the predictions of the model.
    :return: Score of the parameters
    """ 

    f2_score = fbeta_score(y_true, y_pred, beta=2)
    return f2_score



def get_feature_names(column_transformer):
    """Get feature names from all transformers.
    Returns
    -------
    feature_names : list of strings
        Names of the features produced by transform.
    """
    # Remove the internal helper function
    #check_is_fitted(column_transformer)
    
    # Turn loopkup into function for better handling with pipeline later
    def get_names(trans):
        # >> Original get_feature_names() method
        if trans == 'drop' or (
                hasattr(column, '__len__') and not len(column)):
            return []
        if trans == 'passthrough':
            if hasattr(column_transformer, '_df_columns'):
                if ((not isinstance(column, slice))
                        and all(isinstance(col, str) for col in column)):
                    return column
                else:
                    return column_transformer._df_columns[column]
            else:
                indices = np.arange(column_transformer._n_features)
                return ['x%d' % i for i in indices[column]]
        if not hasattr(trans, 'get_feature_names'):
        # >>> Change: Return input column names if no method avaiable
            # Turn error into a warning
            warnings.warn("Transformer %s (type %s) does not "
                                 "provide get_feature_names. "
                                 "Will return input column names if available"
                                 % (str(name), type(trans).__name__))
            # For transformers without a get_features_names method, use the input
            # names to the column transformer
            if column is None:
                return []
            else:
                return [name + "__" + f for f in column]

        return [name + "__" + f for f in trans.get_feature_names()]
    
    ### Start of processing
    feature_names = []
    
    # Allow transformers to be pipelines. Pipeline steps are named differently, so preprocessing is needed
    if type(column_transformer) == sklearn.pipeline.Pipeline:
        l_transformers = [(name, trans, None, None) for step, name, trans in column_transformer._iter()]
    else:
        # For column transformers, follow the original method
        l_transformers = list(column_transformer._iter(fitted=True))
    
    
    for name, trans, column, _ in l_transformers: 
        if type(trans) == sklearn.pipeline.Pipeline:
            # Recursive call on pipeline
            _names = get_feature_names(trans)
            # if pipeline has no transformer that returns names
            if len(_names)==0:
                _names = [name + "__" + f for f in column]
            feature_names.extend(_names)
        else:
            feature_names.extend(get_names(trans))
    
    return feature_names

def evaluate_model(ytest, ypred, ypred_proba = None):
    """   
    Obtain different parameters of the model as ROC-AUC score, Accuracy, Classif report and Conf matrix.
    :param ytest: List with real values.
    :param ypred: List with the predictions of the model.
    :param ypred_proba: List
    :return: Score of the parameters
    """ 
    
    if ypred_proba is not None:
        print('ROC-AUC score of the model: {}'.format(roc_auc_score(ytest, ypred_proba[:, 1])))
    print('Accuracy of the model: {}\n'.format(accuracy_score(ytest, ypred)))
    print('Classification report: \n{}\n'.format(classification_report(ytest, ypred)))
    print('Confusion matrix: \n{}\n'.format(confusion_matrix(ytest, ypred)))

def save_model(clasificador,ruta):
    """   
    Save a pickle file into a directory from the notebook.
    :param clasificador: Classifier
    :param ruta: Local directory
    :return: Backup of a model optimization or a preprocessor
    """
    return pickle.dump(clasificador ,open(ruta, 'wb'))
    

def load_model(ruta):
    """   
    Load a pickle file from a directory into the notebook.
    :param ruta: Local directory
    :return: Backup of a model optimization or a preprocessor
    """
    
    return pickle.load(open(ruta, 'rb'))

def cramers_V(var1,var2) :
    
    """   
    calculate Cramers V statistic for categorial-categorial association.
    :param var1: Categorical variable to compare
    :param var2: Categorical variable to compare
    :return: Value
    """
    crosstab =np.array(pd.crosstab(var1,var2, rownames=None, colnames=None)) 
    stat = chi2_contingency(crosstab)[0] 
    obs = np.sum(crosstab) 
    mini = min(crosstab.shape)-1 
    return (stat/(obs*mini))

def get_deviation_of_mean_perc(df, list_var, target, multiplier):
    
    """
    Returns the percentage of values that exceed the confidence interval.
    :param df: Dataframe to analyze
    :param list_var: List of contionus variables
    :param target: Target variable
    :param multiplier: Number to use as multiplier
    :return: Dataframe
    """
    pd_final = pd.DataFrame()
    
    for i in list_var:
        
        series_mean = df[i].mean()
        series_std = df[i].std()
        std_amp = multiplier * series_std
        left = series_mean - std_amp
        right = series_mean + std_amp
        size_s = df[i].size
        
        perc_goods = df[i][(df[i] >= left) & (df[i] <= right)].size/size_s
        perc_excess = df[i][(df[i] < left) | (df[i] > right)].size/size_s
        
        if perc_excess>0:    
            pd_concat_percent = pd.DataFrame(df[target][(df[i] < left) | (df[i] > right)]\
                                            .value_counts(normalize=True).reset_index()).T
            pd_concat_percent.columns = [pd_concat_percent.iloc[0,0], 
                                         pd_concat_percent.iloc[0,1]]
            pd_concat_percent = pd_concat_percent.drop('index',axis=0)
            pd_concat_percent['Variable'] = i
            pd_concat_percent['Número de outliers'] = df[i][(df[i] < left) | (df[i] > right)].size
            pd_concat_percent['Porcentaje outliers'] = perc_excess
            pd_final = pd.concat([pd_final, pd_concat_percent], axis=0).reset_index(drop=True)
            
    if pd_final.empty:
        print('No existen variables con valores nulos')
        
    return pd_final

def get_corr_matrix(dataset = None, metodo='pearson', size_figure=[12,10]):
    
    '''
    Returns a plot of a correlation matrix using the Pearson method.
    :param dataset: Dataset to analyze
    :param metodo: Method of correlation
    :param size_figure: Configuration of the size of the plot
    :return: Plot
    '''
    if dataset is None:
        print(u'\nHace falta pasar un dataset a la función')
   
    corr = dataset.corr(method=metodo) 
    
    for i in range(corr.shape[0]):
        corr.iloc[i, i] = 0
        
    f, ax = plt.subplots(figsize=size_figure)
    sns.heatmap(corr,annot=True, cmap =plt.cm.Reds, ax = ax )
    ax.set_title('Matriz de Correlación')
    plt.show()     

def list_categorical_variables(df):
    
    '''
    Returns a list with the distinct variables with data type as 'object' in the dataframe.
    :param df: Dataset to analyze
    :return: List
    '''
    list_categorical_variables = list(df.select_dtypes(include=['object']).columns)
    return list_categorical_variables

def unique_values(df, column):
    
    '''
    Returns a list with the distinct values of the variable in the dataframe.
    :param df: Dataset to analyze
    :param column: Objective variable
    :return: List
    '''
    unique_values = df[column].unique().tolist()
    return unique_values


def classes_overview(df=None, obj_val=""):
    
    '''
    Returns a dataframe with percentage and absolute values of the variable.
    :param df: Dataset to analyze
    :param obj_val: Objective variable
    :return: Dataframe
    '''
    temp = df[obj_val].value_counts(normalize=True).mul(100).rename('porcentaje').reset_index()
    temp_cont = df[obj_val].value_counts().reset_index()
    return pd.merge(temp, temp_cont, on=['index'], how='inner')



def plot_feature(df, col_name, target):
    
    """
    Visualize a variable with and without faceting on the loan status.
    - df dataframe
    - col_name is the variable name in the dataframe
    - full_name is the full variable name
    - continuous is True if the variable is continuous, False otherwise
    """
    f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(18,5), dpi=90)
    
    df_n = df.groupby([target, col_name]).size().reset_index()
    df_n = df_n.rename(columns={0:"conteo"})
    df_n["porcentajes"] = df_n['conteo'] / df_n.groupby(target)['conteo'].transform('sum') * 100
      
    sns.countplot(df[col_name], order=sorted(df[col_name].unique()), color='#5975A4', saturation=1, ax=ax1)
    ax1.set_xlabel(col_name)
    ax1.set_ylabel('Count')
    plt.xticks(rotation = 90)
    
    
    sns.barplot(data=df_n, y="porcentajes", x=col_name, hue=target, ax=ax2)
    ax2.set_xlabel(col_name)
    ax2.set_ylabel('Distribución')
    plt.xticks(rotation = 90)
    plt.tight_layout()
    
    

def plot_histograms(df, columns):
    
    """
    Diplay an histogram of a variable from a dataframe..
    :param df: Dataframe with all data
    :param columns: Variable from the df to be plotted
    :return: Histogram plot
    """
    
    k = len(df.columns)
    n = columns
    m = (k - 1) // n + 1
    fig, axes = plt.subplots(m, n, figsize=(n * 5, m * 3))

   
    for i, (name, col) in enumerate(df.iteritems()):
        r, c = i // n, i % n
        ax = axes[r, c]
        col.hist(ax=ax)
        ax2 = col.plot.kde(ax=ax, secondary_y=True, title=name)
        ax2.set_ylim(0)
        
    fig.tight_layout()


def get_outliers(df):

    """
    
    This function get the outliers of a dataframe.

    :param df: Dataframe with the business information.
    :return: Dataframe with the outliers.

    """

    q1=df.quantile(0.25)

    q3=df.quantile(0.75)

    IQR=q3-q1

    outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]

    return outliers