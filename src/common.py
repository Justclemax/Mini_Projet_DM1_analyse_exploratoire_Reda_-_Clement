import functools
import pandas as pd
from typing import List, Tuple, Any, Callable
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


