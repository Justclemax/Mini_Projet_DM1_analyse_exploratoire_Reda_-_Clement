"""
Application Streamlit - Prédiction de Souscription aux Dépôts à Terme
Interface interactive pour tester le modèle KNN
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="Prédiction Marketing Bancaire",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CHARGEMENT DU MODÈLE
# ============================================================================
@st.cache_resource
def load_model():
    """Charger le modèle et tous les objets nécessaires"""
    try:
        model_package = joblib.load('data/models/knn_model_complete.pkl')
        return model_package
    except FileNotFoundError:
        st.error("⚠️ Modèle non trouvé! Veuillez d'abord entraîner le modèle avec train_model.py")
        st.stop()


# Charger le modèle
model_package = load_model()
model = model_package['model']
scaler = model_package['scaler']
label_encoders = model_package['label_encoders']
feature_names = model_package['feature_names']
metrics = model_package['metrics']

# ============================================================================
# EN-TÊTE
# ============================================================================
st.markdown('<div class="main-header">🏦 Prédiction de Souscription aux Dépôts à Terme</div>',
            unsafe_allow_html=True)

st.markdown("""
### 🎯 Objectif
Cette application permet de prédire si un client souscrira à un dépôt à terme 
en fonction de son profil et de ses interactions avec la banque.
""")

# ============================================================================
# SIDEBAR - INFORMATIONS SUR LE MODÈLE
# ============================================================================
with st.sidebar:
    st.header("📊 Informations sur le Modèle")

    st.metric("Modèle", "K-Nearest Neighbors")
    st.metric("Nombre de voisins (k)", model_package['best_k'])

    st.markdown("---")
    st.subheader("🎯 Performances")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
        st.metric("Precision", f"{metrics['precision'] * 100:.1f}%")
    with col2:
        st.metric("Recall", f"{metrics['recall'] * 100:.1f}%")
        st.metric("F1-Score", f"{metrics['f1_score']:.3f}")

    st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")

    st.markdown("---")
    st.markdown("""
    ### 📖 Guide d'utilisation
    1. Remplissez les informations du client
    2. Cliquez sur "Prédire"
    3. Consultez le résultat et la probabilité
    """)

# ============================================================================
# ONGLETS PRINCIPAUX
# ============================================================================
tab1, tab2, tab3 = st.tabs(["🔮 Prédiction Individuelle", "📊 Prédictions en Lot", "📈 Analyse du Modèle"])

# ============================================================================
# ONGLET 1: PRÉDICTION INDIVIDUELLE
# ============================================================================
with tab1:
    st.header("🔮 Prédiction pour un Client Individuel")

    # Formulaire de saisie
    with st.form("prediction_form"):
        st.subheader("Informations Démographiques")
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Âge", min_value=18, max_value=100, value=35, step=1)
            metier = st.selectbox("Profession",
                                  ['cadre', 'technicien', 'entrepreneur', 'ouvrier', 'services',
                                   'retraité', 'employé admin', 'étudiant', 'femme/homme au foyer',
                                   'indépendant', 'au chômage', 'inconnu'])

        with col2:
            etat_civil = st.selectbox("État Civil", ['marié', 'célibataire', 'divorcé'])
            niveau_education = st.selectbox("Niveau d'Éducation",
                                            ['supérieur', 'secondaire', 'primaire', 'inconnu'])

        with col3:
            defaut_credit = st.selectbox("Crédit en Défaut?", ['non', 'oui'])
            solde_annuel_moyen = st.number_input("Solde Annuel Moyen (€)",
                                                 min_value=-10000, max_value=100000,
                                                 value=1500, step=100)

        st.markdown("---")
        st.subheader("Informations Bancaires")
        col4, col5, col6 = st.columns(3)

        with col4:
            pret_immobilier = st.selectbox("Prêt Immobilier?", ['non', 'oui'])
            pret_personel = st.selectbox("Prêt Personnel?", ['non', 'oui'])

        with col5:
            type_contact = st.selectbox("Type de Contact", ['mobile', 'téléphone', 'inconnu'])
            jour = st.number_input("Jour du Contact", min_value=1, max_value=31, value=15)

        with col6:
            mois = st.selectbox(
                "Mois du Contact",
                ['janvier', 'février', 'mars', 'avril', 'mai', 'juin',
                 'juillet', 'août', 'septembre', 'octobre', 'novembre', 'décembre']
            )

            duree_appel = st.number_input(
                "Durée de l'Appel (secondes)",
                min_value=0,
                max_value=5000,
                value=180,
                step=10
            )

        st.markdown("---")
        st.subheader("Historique de Campagne")
        col7, col8, col9 = st.columns(3)

        with col7:
            nb_appels_campagne = st.number_input("Nombre d'Appels (Campagne Actuelle)",
                                                 min_value=1, max_value=50, value=2, step=1)

        with col8:
            jours_depuis_dernier = st.number_input("Jours Depuis Dernier Contact",
                                                   min_value=-1, max_value=999, value=-1, step=1,
                                                   help="-1 si jamais contacté")

        with col9:
            appels_precedents = st.number_input("Nombre d'Appels (Campagnes Précédentes)",
                                                min_value=0, max_value=50, value=0, step=1)

        resultat_prec = st.selectbox("Résultat Campagne Précédente",
                                     ['inconnu', 'échec', 'succès', 'autre'])

        # Bouton de soumission
        st.markdown("---")
        submit_button = st.form_submit_button("🚀 Prédire", use_container_width=True)

    # Prédiction
    if submit_button:
        # Créer le dictionnaire de données
        input_data = {
            'age': age,
            'metier': metier,
            'etat_civil': etat_civil,
            'niveau_education': niveau_education,
            'defaut_credit': defaut_credit,
            'solde_annuel_moyen': solde_annuel_moyen,
            'pret_immobilier': pret_immobilier,
            'pret_personel': pret_personel,
            'type_contact': type_contact,
            'jour': jour,
            'mois': mois,
            'duree_appel': duree_appel,
            'nb_appels_campagne': nb_appels_campagne,
            'jours_depuis_dernier': jours_depuis_dernier,
            'appels_precedents': appels_precedents,
            'resultat_prec': resultat_prec
        }

        # Créer un DataFrame
        input_df = pd.DataFrame([input_data])

        # Encoder les variables catégorielles
        for col, le in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = le.transform(input_df[col])
                except ValueError:
                    st.error(f"⚠️ Valeur inconnue pour {col}. Utilisation de la valeur par défaut.")
                    input_df[col] = 0

        # Normaliser
        input_scaled = scaler.transform(input_df)

        # Prédire
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        # Afficher les résultats
        st.markdown("---")
        st.subheader("📊 Résultats de la Prédiction")

        col_res1, col_res2, col_res3 = st.columns([1, 2, 1])

        with col_res2:
            if prediction == 1:
                st.markdown("""
                <div class="success-box">
                    <h2 style='text-align: center; color: #155724;'>✅ SOUSCRIPTION PROBABLE</h2>
                    <p style='text-align: center; font-size: 1.2rem;'>
                        Ce client est susceptible de souscrire au dépôt à terme
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-box">
                    <h2 style='text-align: center; color: #856404;'>❌ SOUSCRIPTION IMPROBABLE</h2>
                    <p style='text-align: center; font-size: 1.2rem;'>
                        Ce client est peu susceptible de souscrire au dépôt à terme
                    </p>
                </div>
                """, unsafe_allow_html=True)

        # Graphique de probabilité
        st.subheader("🎯 Probabilités")

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=['Non', 'Oui'],
            y=[probability[0] * 100, probability[1] * 100],
            marker_color=['#e74c3c', '#2ecc71'],
            text=[f'{probability[0] * 100:.1f}%', f'{probability[1] * 100:.1f}%'],
            textposition='auto',
            textfont=dict(size=16, color='white', family='Arial Black')
        ))

        fig.update_layout(
            title='Probabilité de Souscription',
            xaxis_title='Souscription',
            yaxis_title='Probabilité (%)',
            yaxis_range=[0, 100],
            height=400,
            showlegend=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # Recommandations
        st.subheader("💡 Recommandations")

        if prediction == 1:
            if probability[1] > 0.7:
                st.success("🎯 **Priorité élevée** - Forte probabilité de conversion. Contactez ce client rapidement!")
            elif probability[1] > 0.5:
                st.info("📞 **Priorité moyenne** - Probabilité modérée. Préparez une offre personnalisée.")
            else:
                st.warning("🤔 **Priorité faible** - Probabilité limite. Considérez le contexte avant de contacter.")
        else:
            if duree_appel < 180:
                st.info("💡 Conseil: Augmenter la durée des appels pourrait améliorer les chances.")
            if nb_appels_campagne > 3:
                st.warning("⚠️ Attention: Trop de contacts peuvent être contre-productifs.")
            if resultat_prec == 'échec':
                st.info("📝 Note: L'historique négatif influence la prédiction. Nouvelle approche recommandée.")

# ============================================================================
# ONGLET 2: PRÉDICTIONS EN LOT
# ============================================================================
with tab2:
    st.header("📊 Prédictions en Lot (Fichier CSV)")

    st.markdown("""
    ### 📋 Instructions
    1. Téléchargez le modèle de fichier CSV
    2. Remplissez-le avec les données de vos clients
    3. Uploadez le fichier pour obtenir les prédictions
    """)

    # Télécharger le template
    template_data = {
        'age': [35, 42],
        'metier': ['cadre', 'technicien'],
        'etat_civil': ['marié', 'célibataire'],
        'niveau_education': ['supérieur', 'secondaire'],
        'defaut_credit': ['non', 'non'],
        'solde_annuel_moyen': [1500, 2000],
        'pret_immobilier': ['oui', 'non'],
        'pret_personel': ['non', 'oui'],
        'type_contact': ['téléphone portable', 'téléphone'],
        'jour': [15, 20],
        'mois': ['mai', 'décembre'],
        'duree_appel': [180, 250],
        'nb_appels_campagne': [2, 1],
        'jours_depuis_dernier': [-1, 180],
        'appels_precedents': [0, 1],
        'resultat_prec': ['inconnu', 'succès']
    }

    template_df = pd.DataFrame(template_data)

    csv = template_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger le Modèle CSV",
        data=csv,
        file_name="template_clients.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Upload de fichier
    uploaded_file = st.file_uploader("📤 Charger un fichier CSV", type=['csv'])

    if uploaded_file is not None:
        try:
            # Lire le fichier
            df_batch = pd.read_csv(uploaded_file)

            st.success(f"✅ Fichier chargé: {len(df_batch)} clients")

            # Afficher un aperçu
            with st.expander("👀 Aperçu des données"):
                st.dataframe(df_batch.head())

            # Bouton pour prédire
            if st.button("🚀 Générer les Prédictions", use_container_width=True):
                with st.spinner("Calcul en cours..."):
                    # Encoder
                    df_encoded = df_batch.copy()
                    for col, le in label_encoders.items():
                        if col in df_encoded.columns:
                            try:
                                df_encoded[col] = le.transform(df_encoded[col])
                            except:
                                df_encoded[col] = 0

                    # Normaliser
                    X_batch = scaler.transform(df_encoded)

                    # Prédire
                    predictions = model.predict(X_batch)
                    probabilities = model.predict_proba(X_batch)[:, 1]

                    # Ajouter les résultats
                    df_batch['prediction'] = ['Oui' if p == 1 else 'Non' for p in predictions]
                    df_batch['probabilite_souscription'] = (probabilities * 100).round(2)
                    df_batch['priorite'] = pd.cut(
                        probabilities,
                        bins=[0, 0.3, 0.6, 1.0],
                        labels=['Faible', 'Moyenne', 'Élevée']
                    )

                    # Afficher les résultats
                    st.subheader("📊 Résultats des Prédictions")

                    # Statistiques
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Clients", len(df_batch))
                    with col2:
                        n_yes = (predictions == 1).sum()
                        st.metric("Souscriptions Probables", n_yes)
                    with col3:
                        taux = n_yes / len(df_batch) * 100
                        st.metric("Taux de Conversion Prédit", f"{taux:.1f}%")
                    with col4:
                        prob_moyenne = probabilities.mean() * 100
                        st.metric("Probabilité Moyenne", f"{prob_moyenne:.1f}%")

                    # Tableau des résultats
                    st.dataframe(
                        df_batch.style.background_gradient(
                            subset=['probabilite_souscription'],
                            cmap='RdYlGn'
                        ),
                        height=400
                    )

                    # Télécharger les résultats
                    result_csv = df_batch.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        label="📥 Télécharger les Résultats",
                        data=result_csv,
                        file_name="predictions_clients.csv",
                        mime="text/csv",
                        use_container_width=True
                    )

                    # Graphique de distribution
                    fig = px.histogram(
                        df_batch,
                        x='probabilite_souscription',
                        color='prediction',
                        nbins=20,
                        title='Distribution des Probabilités de Souscription',
                        color_discrete_map={'Non': '#e74c3c', 'Oui': '#2ecc71'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"❌ Erreur lors du traitement: {str(e)}")

# ============================================================================
# ONGLET 3: ANALYSE DU MODÈLE
# ============================================================================
with tab3:
    st.header("📈 Analyse du Modèle KNN")

    # Métriques détaillées
    st.subheader("🎯 Métriques de Performance")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%",
                  help="Pourcentage de prédictions correctes")
    with col2:
        st.metric("Precision", f"{metrics['precision'] * 100:.2f}%",
                  help="Proportion de vrais positifs parmi les prédictions positives")
    with col3:
        st.metric("Recall", f"{metrics['recall'] * 100:.2f}%",
                  help="Proportion de vrais positifs détectés")
    with col4:
        st.metric("F1-Score", f"{metrics['f1_score']:.3f}",
                  help="Moyenne harmonique de la précision et du rappel")
    with col5:
        st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}",
                  help="Aire sous la courbe ROC")

    # Informations sur l'entraînement
    st.markdown("---")
    st.subheader("ℹ️ Informations sur l'Entraînement")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info(f"**Échantillons d'entraînement:** {model_package['training_info']['train_samples']:,}")
    with col2:
        st.info(f"**Échantillons de test:** {model_package['training_info']['test_samples']:,}")
    with col3:
        st.info(f"**Nombre de features:** {model_package['training_info']['n_features']}")

    # Importance des variables
    st.markdown("---")
    st.subheader("📊 Variables les Plus Importantes")

    # Charger l'importance des features si disponible
    try:
        feature_importance = pd.read_csv('../models/feature_importance.csv').head(10)

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=feature_importance['feature'],
            x=feature_importance['importance'],
            orientation='h',
            marker_color='steelblue',
            text=feature_importance['importance'].round(4),
            textposition='auto'
        ))

        fig.update_layout(
            title='Top 10 Variables les Plus Importantes',
            xaxis_title='Importance (Permutation)',
            yaxis_title='Variable',
            height=500,
            yaxis={'categoryorder': 'total ascending'}
        )

        st.plotly_chart(fig, use_container_width=True)

    except FileNotFoundError:
        st.warning("⚠️ Données d'importance des variables non disponibles")

    # Interprétation
    st.markdown("---")
    st.subheader("💡 Interprétation des Résultats")

    st.markdown("""
    #### 🔍 Points Clés

    - **Durée d'appel**: Variable la plus importante - plus l'appel est long, plus la probabilité de souscription augmente
    - **Résultat campagne précédente**: Un succès passé est un fort indicateur de succès futur
    - **Nombre de contacts**: Attention à ne pas sur-contacter les clients (effet contre-productif)
    - **Caractéristiques démographiques**: Âge, profession et éducation jouent un rôle modéré

    #### 📈 Recommandations Stratégiques

    1. **Qualité > Quantité**: Privilégier des appels de qualité plutôt que multiplier les contacts
    2. **Ciblage intelligent**: Prioriser les clients avec un historique positif
    3. **Formation des agents**: Investir dans la formation pour améliorer la durée et qualité des appels
    4. **Segmentation**: Adapter l'approche selon le profil démographique et professionnel
    """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 2rem;'>
    <p>🏦 Application de Prédiction Marketing Bancaire</p>
    <p>Développée avec Streamlit • Modèle: K-Nearest Neighbors</p>
</div>
""", unsafe_allow_html=True)
