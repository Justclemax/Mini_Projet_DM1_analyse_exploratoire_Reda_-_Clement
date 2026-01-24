import pandas as pd
import numpy as np

from scipy.stats import (
    f_oneway,
    chi2_contingency,
    pearsonr,
    spearmanr
)

import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.mediation import Mediation


class StatisticalTests:
    """
    Classe centralisée pour les tests statistiques
    orientés analyse décisionnelle.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    # -------------------------------------------------
    # ANOVA
    # -------------------------------------------------
    def anova(self, numeric_var: str, factor: str):
        """
        ANOVA à un facteur
        """
        groups = [
            self.df[self.df[factor] == g][numeric_var]
            for g in self.df[factor].unique()
        ]
        f_stat, p_value = f_oneway(*groups)
        return {
            "test": "ANOVA",
            "variable": numeric_var,
            "factor": factor,
            "F": f_stat,
            "p_value": p_value
        }

    # -------------------------------------------------
    # MANOVA
    # -------------------------------------------------
    def manova(self, dependent_vars: list, factor: str):
        """
        MANOVA
        """
        formula = f"{' + '.join(dependent_vars)} ~ {factor}"
        model = MANOVA.from_formula(formula, data=self.df)
        return model.mv_test()

    # -------------------------------------------------
    # ANCOVA / MANCOVA
    # -------------------------------------------------
    def ancova(self, dependent: str, factor: str, covariates: list):
        """
        ANCOVA (1 variable dépendante)
        """
        formula = f"{dependent} ~ {factor} + {' + '.join(covariates)}"
        model = smf.ols(formula, data=self.df).fit()
        return sm.stats.anova_lm(model, typ=2)

    def mancova(self, dependents: list, factor: str, covariates: list):
        """
        MANCOVA
        """
        formula = (
            f"{' + '.join(dependents)} ~ {factor} + {' + '.join(covariates)}"
        )
        model = MANOVA.from_formula(formula, data=self.df)
        return model.mv_test()

    # -------------------------------------------------
    # KHI-DEUX
    # -------------------------------------------------
    def chi2(self, var1: str, var2: str):
        """
        Test du Khi-deux
        """
        table = pd.crosstab(self.df[var1], self.df[var2])
        chi2, p, dof, _ = chi2_contingency(table)
        return {
            "test": "Chi2",
            "var1": var1,
            "var2": var2,
            "chi2": chi2,
            "dof": dof,
            "p_value": p
        }

    # -------------------------------------------------
    # CORRÉLATION
    # -------------------------------------------------
    def correlation(self, var1: str, var2: str, method="pearson"):
        """
        Corrélation Pearson ou Spearman
        """
        if method == "pearson":
            corr, p = pearsonr(self.df[var1], self.df[var2])
        elif method == "spearman":
            corr, p = spearmanr(self.df[var1], self.df[var2])
        else:
            raise ValueError("method must be 'pearson' or 'spearman'")

        return {
            "method": method,
            "var1": var1,
            "var2": var2,
            "correlation": corr,
            "p_value": p
        }

    # -------------------------------------------------
    # MÉDIATION – API OFFICIELLE STATSMODELS
    # -------------------------------------------------
    def mediation(
        self,
        x: str,
        mediator: str,
        y: str,
        covariates: list | None = None,
        n_rep: int = 1000
    ):
        """
        Médiation basée sur statsmodels.stats.mediation.Mediation
        avec bootstrap.
        """

        cov = ""
        if covariates:
            cov = " + " + " + ".join(covariates)

        # Modèle du médiateur
        mediator_model = smf.ols(
            f"{mediator} ~ {x}{cov}",
            data=self.df
        )

        # Modèle du résultat
        outcome_model = smf.ols(
            f"{y} ~ {x} + {mediator}{cov}",
            data=self.df
        )

        mediation = Mediation(
            outcome_model,
            mediator_model,
            exposure=x,
            mediator=mediator
        )

        result = mediation.fit(
            n_rep=n_rep,
            method="bootstrap"
        )

        return result.summary()

    # -------------------------------------------------
    # MATRICE DES P-VALUES (corrélations)
    # -------------------------------------------------
    def p_value_matrix(self, numeric_vars: list, method="pearson"):
        """
        Matrice des p-values de corrélation
        """
        matrix = pd.DataFrame(
            np.ones((len(numeric_vars), len(numeric_vars))),
            index=numeric_vars,
            columns=numeric_vars
        )

        for i in numeric_vars:
            for j in numeric_vars:
                if i != j:
                    if method == "pearson":
                        _, p = pearsonr(self.df[i], self.df[j])
                    else:
                        _, p = spearmanr(self.df[i], self.df[j])
                    matrix.loc[i, j] = p

        return matrix