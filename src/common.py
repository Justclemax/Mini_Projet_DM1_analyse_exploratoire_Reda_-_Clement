from typing import Any, Callable
import functools
import pandas as pd


def log_outliers(func: Callable) -> Callable:
    """Décorateur pour enregistrer le nombre d'outliers supprimés par la fonction."""

    @functools.wraps(func)
    def wrapper(df: pd.DataFrame, column: Any, *args, **kwargs) -> pd.DataFrame:
        initial_shape = df.shape[0]
        # Appel de la fonction de filtrage
        df_clean = func(df, column, *args, **kwargs)
        final_shape = df_clean.shape[0]

        diff = initial_shape - final_shape
        print(f"📊 [PROCESS] Variable '{column}' : {diff} outliers supprimés ({initial_shape} -> {final_shape})")
        return df_clean

    return wrapper


@log_outliers
def remove_outliers_iqr(df: pd.DataFrame, column: Any) -> pd.DataFrame:
    """
    Supprime les valeurs aberrantes d'un DataFrame selon la méthode de l'Intervalle Interquartile (IQR).

    Args:
        df (pd.DataFrame): Le DataFrame contenant les données bancaires.
        column (Any): Le nom de la colonne sur laquelle appliquer le filtre (ex: 'solde_annuel').

    Returns:
        pd.DataFrame: Un nouveau DataFrame filtré sans les valeurs extrêmes.
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Filtrage précis
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]


from typing import List, Tuple
import pandas as pd


def identify_features(df: pd.DataFrame, target: str = "cible", threshold: int = 10) -> Tuple[List[str], List[str]]:
    """
    Identifie et sépare les variables catégorielles et continues d'un DataFrame.

    Args:
        df: Le DataFrame à analyser.
        target: Le nom de la variable cible à exclure.
        threshold: Nombre max de valeurs uniques pour considérer une variable numérique comme catégorielle.

    Returns:
        Tuple contenant (liste_catégorielles, liste_continues).
    """
    # 1. On exclut la cible dès le départ
    features = df.drop(columns=[target]) if target in df.columns else df

    # 2. Identification par type de données (Approche hybride)
    # On considère comme catégoriel : les types 'object', 'category' et les entiers < threshold
    categorical = features.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

    # Pour les colonnes numériques, on vérifie la cardinalité (threshold)
    numerical_cols = features.select_dtypes(include=['number']).columns

    continuous = []
    for col in numerical_cols:
        if df[col].nunique() < threshold:
            categorical.append(col)
        else:
            continuous.append(col)

    return categorical, continuous


