from typing import Dict, List

import numpy as np
import pandas as pd
from IPython.display import display_html
from itertools import chain, cycle
from sklearn.metrics import confusion_matrix

def compute_confusion_matrix(y_valid: np.ndarray,
                             y_pred_proba: np.array,
                             threshold: float) -> pd.DataFrame:
    """ Computes the confusion matrix given probabilities, threshold and labels

    Parameters
    ----------
    y_valid : np.ndarray
        Target column containing the labels
    y_pred_proba : np.ndarray
        Probabilities columns from machine learning model
    threshold : float
        Threshold for defining 0 or 1 predictions given probabilities

    Returns
    -------
    pd.DataFrame
        Confusion matrix given the threshold
    """

    y_pred = (y_pred_proba >= threshold).astype(bool)
    conf_matrices = confusion_matrix(y_valid, y_pred)
    conf_matrix = pd.DataFrame(conf_matrices, columns=['1', '0'], index=['1', '0'])
    conf_matrix.columns = pd.MultiIndex.from_product([['Prediction'], conf_matrix.columns])
    conf_matrix.index = pd.MultiIndex.from_product([['True'], conf_matrix.index])

    return conf_matrix

def compute_tpr_fpr(confusion_matrices: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """ Computes the TPR and FPR metrics given the confusion matrices

    Parameters
    ----------
    confusion_matrices : Dict[str: pd.DataFrame]
        Dict containing different confusion matrices        

    Returns
    -------
    pd.DataFrame
        Confusion matrix given the threshold
    """

    df_metrics = pd.DataFrame({}, columns=['Threshold', 'TPR', 'FPR'])
    for key, df in confusion_matrices.items():
        arr = df.values
        tpr = arr[0][0] / (arr[0][0] + arr[0][1])
        fpr = arr[1][0] / (arr[1][0] + arr[1][1])
        dict_to_df = {'Threshold': key, 'TPR': tpr, 'FPR': fpr}
        df_metrics = pd.concat([df_metrics, pd.DataFrame(dict_to_df, index=[0])])
        
    return df_metrics.round(3).reset_index(drop=True)

def display_side_by_side(*args: pd.DataFrame,
                         titles: List[str] = cycle([''])) -> None:
    """Displays two or more dataframes on Jupyter

    Parameters
    ----------
    args : List[pd.DataFrame]
        List containing DataFrames
    titles : List
        List containing the titles of the DataFrames

    Returns
    -------
    None
    
    Acknowledgement
    ---------------
    Special thanks to: @ntg @Antony_Hatchkins et al. on StackOverflow for this piece of code
    https://stackoverflow.com/questions/38783027/jupyter-notebook-display-two-pandas-tables-side-by-side
    """

    html_str = ''
    for df, title in zip(args, chain(titles, cycle(['</br>']))):
        html_str += '<th style="text-align:center"><td style="vertical-align:top">'
        html_str += f'<h2>{title}</h2>'
        html_str += df.to_html().replace('table', 'table style="display:inline"')
        html_str += '</td></th>'
    display_html(html_str, raw=True)

    return



