import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ============================================================================
# SECRETS — Lecture via st.secrets (valeurs définies dans .streamlit/secrets.toml)
# Exemple d'utilisation (décommenter si les clés sont configurées) :
# api_key     = st.secrets["api_keys"]["openai"]
# db_password = st.secrets["database"]["password"]
# ============================================================================

# ============================================================================
# TODO 1 : Configuration de la page
# ============================================================================
st.set_page_config(
    page_title="Prédiction d'Approbation de Prêt",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# TODO 2 : Titre et description
st.title("🏦 Prédiction d'Approbation de Prêt")
st.markdown("---")

# ============================================================================
# TODO 3 : Sidebar - Sélection du modèle
# ============================================================================
st.sidebar.header("⚙️ Configuration")
model_choice = st.sidebar.selectbox(
    "Choisissez un modèle",
    ["Régression Logistique"]
)

# ============================================================================
# TODO 4 : Sidebar - Info sur le modèle
# ============================================================================
st.sidebar.info("📊 Modèle linéaire, interprétable")

# ============================================================================
# TODO 5 : Sidebar - Section "À propos"
# ============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📖 À propos")
st.sidebar.markdown(
    "Cette application prédit l'approbation d'une demande de prêt "
    "à partir de données clients grâce à des modèles de Machine Learning."
)

# ============================================================================
# TODO 6 : Fonction de chargement des données avec cache
# ============================================================================
@st.cache_data
def load_data():
    """Charge les données depuis le fichier CSV"""
    try:
        df = pd.read_csv("loancleaned.csv", index_col=0)
        return df
    except FileNotFoundError:
        st.error("❌ Fichier 'loancleaned.csv' introuvable.")
        return None

# ============================================================================
# TODO 7 : Fonction de chargement du modèle avec cache
# ============================================================================
@st.cache_resource
def load_model(model_name):
    """Charge le modèle sélectionné"""
    try:
        model = joblib.load("modele_logistic_regression.pkl")
        try:
            scaler = joblib.load("models/scaler.pkl")
        except FileNotFoundError:
            scaler = None
        return model, scaler
    except FileNotFoundError:
        st.error(f"❌ Modèle '{model_name}' introuvable.")
        return None, None

# ============================================================================
# TODO 8 : Charger les données
# ============================================================================
df = load_data()

# ============================================================================
# TODO 9 : Créer les onglets
# ============================================================================
if df is not None:

    tab1, tab2, tab3 = st.tabs(["📊 Exploration", "🔮 Prédiction", "📈 Performance"])

    # ========================================================================
    # TODO 10 : Contenu de l'onglet Exploration
    # ========================================================================
    with tab1:
        st.header("📊 Exploration des Données")
        st.markdown("Visualisez et explorez le jeu de données utilisé pour l'entraînement.")
        st.info("💡 Utilisez les filtres ci-dessous pour affiner l'exploration. Les graphiques se mettent à jour automatiquement.")

        # TODO 1 : Métriques KPIs
        st.subheader("📊 Indicateurs Clés")
        col1, col2, col3, col4 = st.columns(4)
        taux_approbation = (df["Loan_Status"] == 1).mean() * 100
        col1.metric("Total demandes", len(df))
        col2.metric("Taux d'approbation", f"{taux_approbation:.1f}%")
        col3.metric("Montant moyen prêt", f"{df['LoanAmount'].mean():.0f} €")
        col4.metric("Revenu moyen", f"{df['ApplicantIncome'].mean():.0f} €")

        st.markdown("---")

        # TODO 7 : Filtres interactifs
        st.subheader("🔍 Filtres Interactifs")
        fcol1, fcol2 = st.columns(2)
        with fcol1:
            min_income = int(df["ApplicantIncome"].min())
            max_income = int(df["ApplicantIncome"].max())
            income_range = st.slider(
                "Filtrer par revenu (ApplicantIncome)",
                min_value=min_income, max_value=max_income,
                value=(min_income, max_income)
            )
        with fcol2:
            edu_labels = {0: "Non diplômé", 1: "Diplômé"}
            selected_education = st.multiselect(
                "Niveau d'éducation",
                options=df["Education"].unique(),
                default=list(df["Education"].unique()),
                format_func=lambda x: edu_labels[x]
            )

        df_filtered = df[
            (df["ApplicantIncome"].between(*income_range)) &
            (df["Education"].isin(selected_education if selected_education else df["Education"].unique()))
        ]

        st.caption(f"{len(df_filtered)} observations affichées")
        st.markdown("---")

        # TODO 2 : Histogramme distribution des revenus
        st.subheader("📈 Distribution des Revenus")
        fig_hist = px.histogram(
            df_filtered, x="ApplicantIncome", nbins=30,
            title="Distribution des Revenus des Demandeurs",
            labels={"ApplicantIncome": "Revenu (€)", "count": "Nombre"}
        )
        fig_hist.add_vline(
            x=df_filtered["ApplicantIncome"].mean(),
            line_dash="dash", line_color="red",
            annotation_text=f"Moyenne: {df_filtered['ApplicantIncome'].mean():.0f} €"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # TODO 3 : Box plot montants de prêt
        st.subheader("📦 Montants de Prêt Demandés")
        fig_box = px.box(
            df_filtered, y="LoanAmount",
            title="Distribution des Montants de Prêt",
            labels={"LoanAmount": "Montant du Prêt (€)"}
        )
        median = df_filtered["LoanAmount"].median()
        q1 = df_filtered["LoanAmount"].quantile(0.25)
        q3 = df_filtered["LoanAmount"].quantile(0.75)
        fig_box.add_annotation(x=0.3, y=median, text=f"Médiane: {median:.0f}", showarrow=False)
        fig_box.add_annotation(x=0.3, y=q1, text=f"Q1: {q1:.0f}", showarrow=False)
        fig_box.add_annotation(x=0.3, y=q3, text=f"Q3: {q3:.0f}", showarrow=False)
        st.plotly_chart(fig_box, use_container_width=True)

        gcol1, gcol2 = st.columns(2)

        # TODO 4 : Bar chart taux d'approbation par éducation
        with gcol1:
            st.subheader("🎓 Approbation selon l'Éducation")
            edu_group = df_filtered.groupby(["Education", "Loan_Status"]).size().reset_index(name="Count")
            edu_total = df_filtered.groupby("Education")["Loan_Status"].count().reset_index(name="Total")
            edu_group = edu_group.merge(edu_total, on="Education")
            edu_group["Percentage"] = edu_group["Count"] / edu_group["Total"] * 100
            edu_approved = edu_group[edu_group["Loan_Status"] == 1].copy()
            edu_approved["Education"] = edu_approved["Education"].map({0: "Non diplômé", 1: "Diplômé"})
            fig_bar = px.bar(
                edu_approved, x="Education", y="Percentage",
                title="Taux d'Approbation par Niveau d'Éducation",
                labels={"Percentage": "Taux d'approbation (%)"},
                color="Education"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # TODO 5 : Pie chart répartition approuvé/rejeté
        with gcol2:
            st.subheader("🥧 Répartition Approuvé/Rejeté")
            counts = df_filtered["Loan_Status"].value_counts()
            fig_pie = px.pie(
                values=counts.values,
                names=["Approuvé" if i == 1 else "Rejeté" for i in counts.index],
                title="Répartition des Décisions",
                color_discrete_sequence=["#2ecc71", "#e74c3c"],
                hole=0.4
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        # TODO 6 : Heatmap corrélation
        st.subheader("🔥 Matrice de Corrélation")
        num_cols = df_filtered.select_dtypes(include=["float64", "int64"])
        corr = num_cols.corr()
        fig_heat = go.Figure(go.Heatmap(
            z=corr.values,
            x=corr.columns.tolist(),
            y=corr.columns.tolist(),
            colorscale="RdBu",
            zmid=0
        ))
        fig_heat.update_layout(title="Matrice de Corrélation", height=500)
        st.plotly_chart(fig_heat, use_container_width=True)

        # TODO 8 : Téléchargement CSV
        st.markdown("---")
        st.download_button(
            label="📥 Télécharger les données filtrées (CSV)",
            data=df_filtered.to_csv(index=False).encode("utf-8"),
            file_name="loan_data.csv",
            mime="text/csv"
        )

    # ========================================================================
    # TODO 11 : Contenu de l'onglet Prédiction
    # ========================================================================
    with tab2:
        st.header("🔮 Prédiction")
        st.markdown("Saisissez les informations d'un client pour prédire l'approbation de son prêt.")
        st.info("💡 Remplissez tous les champs du formulaire puis cliquez sur **Prédire** pour obtenir une décision instantanée.")

        model, scaler = load_model(model_choice)

        if model is None:
            st.error("❌ Impossible de charger le modèle.")
        else:
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)

                with col1:
                    st.subheader("💰 Informations financières")
                    applicant_income = st.number_input("Revenu du demandeur (€)", min_value=0, value=5000, step=100)
                    coapplicant_income = st.number_input("Revenu du co-demandeur (€)", min_value=0, value=0, step=100)
                    loan_amount = st.number_input("Montant du prêt (€)", min_value=1, value=128, step=1)
                    loan_term = st.selectbox("Durée du prêt (mois)", options=[12, 36, 60, 84, 120, 180, 240, 300, 360, 480], index=9)

                with col2:
                    st.subheader("👤 Informations personnelles")
                    gender = st.selectbox("Genre", options=["Homme", "Femme"])
                    married = st.selectbox("Statut marital", options=["Marié(e)", "Célibataire"])
                    dependents = st.selectbox("Nombre de personnes à charge", options=[0, 1, 2, 3])
                    education = st.selectbox("Niveau d'éducation", options=["Diplômé(e)", "Non diplômé(e)"])
                    self_employed = st.selectbox("Travailleur indépendant", options=["Non", "Oui"])
                    credit_history = st.selectbox("Historique de crédit", options=["Bon (1)", "Mauvais (0)"])
                    area = st.selectbox("Zone géographique", options=["Urban", "Semiurban", "Rural"])

                submitted = st.form_submit_button("🔍 Prédire", use_container_width=True)

            if submitted:
                # Validations
                warnings = []
                if applicant_income < 1000:
                    warnings.append("⚠️ Revenu très faible (< 1 000 €)")
                if loan_amount > applicant_income * 0.5:
                    warnings.append("⚠️ Montant du prêt élevé par rapport au revenu")
                if credit_history == "Mauvais (0)":
                    warnings.append("⚠️ Mauvais historique de crédit — impact fort sur la décision")
                for w in warnings:
                    st.warning(w)

                # Encodage
                gender_enc     = 1 if gender == "Homme" else 0
                married_enc    = 1 if married == "Marié(e)" else 0
                education_enc  = 1 if education == "Diplômé(e)" else 0
                self_emp_enc   = 1 if self_employed == "Oui" else 0
                credit_enc     = 1 if credit_history == "Bon (1)" else 0
                area_rural     = 1 if area == "Rural" else 0
                area_semiurban = 1 if area == "Semiurban" else 0
                area_urban     = 1 if area == "Urban" else 0

                # Feature engineering
                total_income      = applicant_income + coapplicant_income
                emi               = loan_amount * 1000 / loan_term
                emi_income_ratio  = emi / total_income if total_income > 0 else 0
                log_applicant     = np.log(applicant_income + 1)
                log_coapplicant   = np.log(coapplicant_income + 1)
                log_loan          = np.log(loan_amount + 1)
                log_total         = np.log(total_income + 1)
                has_coapplicant   = 1 if coapplicant_income > 0 else 0
                high_loan         = 1 if loan_amount > 128 else 0

                # Vecteur de features (ordre exact du modèle)
                input_data = pd.DataFrame([{
                    "Unnamed: 0":           0,
                    "Gender":               gender_enc,
                    "Married":              married_enc,
                    "Dependents":           dependents,
                    "Education":            education_enc,
                    "Self_Employed":        self_emp_enc,
                    "ApplicantIncome":      applicant_income,
                    "CoapplicantIncome":    coapplicant_income,
                    "LoanAmount":           loan_amount,
                    "Loan_Amount_Term":     loan_term,
                    "Credit_History":       credit_enc,
                    "TotalIncome":          total_income,
                    "EMI_Income_Ratio":     emi_income_ratio,
                    "Log_ApplicantIncome":  log_applicant,
                    "Log_CoapplicantIncome":log_coapplicant,
                    "Log_LoanAmount":       log_loan,
                    "Log_TotalIncome":      log_total,
                    "Has_Coapplicant":      has_coapplicant,
                    "HighLoan":             high_loan,
                    "Area_Rural":           area_rural,
                    "Area_Semiurban":       area_semiurban,
                    "Area_Urban":           area_urban,
                }])

                prediction = model.predict(input_data)[0]
                proba      = model.predict_proba(input_data)[0][1]

                st.markdown("---")
                st.subheader("📋 Résultat")

                if prediction == 1:
                    st.success(f"✅ Prêt **APPROUVÉ** avec une probabilité de {proba*100:.1f}%")
                else:
                    st.error(f"❌ Prêt **REFUSÉ** avec une probabilité d'approbation de {proba*100:.1f}%")

                st.markdown("**Probabilité d'approbation**")
                st.progress(float(proba))
                st.caption(f"{proba*100:.1f}%")

                # Section Explication — Top 5 features influentes
                st.markdown("---")
                st.subheader("🔍 Explication de la décision")
                st.caption("Top 5 des variables les plus influentes (coefficients du modèle × valeur)")

                coefs    = model.coef_[0]
                features = model.feature_names_in_
                contributions = {f: abs(c * input_data[f].values[0]) for f, c in zip(features, coefs)}
                top5 = sorted(contributions.items(), key=lambda x: x[1], reverse=True)[:5]

                top5_df = pd.DataFrame(top5, columns=["Feature", "Importance"])
                fig_exp = px.bar(
                    top5_df, x="Importance", y="Feature",
                    orientation="h",
                    title="Top 5 variables influentes",
                    color="Importance",
                    color_continuous_scale="Blues"
                )
                fig_exp.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig_exp, use_container_width=True)

    # ========================================================================
    # TODO 12 : Contenu de l'onglet Performance
    # ========================================================================
    with tab3:
        st.header("📈 Performance du Modèle")
        st.markdown(f"Évaluation des performances du modèle **{model_choice}**.")
        st.info("💡 Les métriques sont calculées sur **20 %** des données (jeu de test), jamais vus pendant l'entraînement.")

        model, scaler = load_model(model_choice)

        if model is not None:
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import (accuracy_score, precision_score,
                                         recall_score, f1_score,
                                         roc_auc_score, confusion_matrix, roc_curve)

            # Préparer les données de test
            X = df.drop("Loan_Status", axis=1)
            X["Unnamed: 0"] = range(len(X))
            y = df["Loan_Status"]
            _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            if hasattr(model, "feature_names_in_"):
                X_test = X_test[model.feature_names_in_]

            y_pred  = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            # --- Métriques globales ---
            st.subheader("📊 Métriques Globales")
            st.caption("Cliquez sur les expanders en bas pour une explication détaillée de chaque métrique.")
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy",  f"{accuracy_score(y_test, y_pred)*100:.1f}%")
            m2.metric("Precision", f"{precision_score(y_test, y_pred)*100:.1f}%")
            m3.metric("Recall",    f"{recall_score(y_test, y_pred)*100:.1f}%")
            m4.metric("F1-Score",  f"{f1_score(y_test, y_pred)*100:.1f}%")
            m5.metric("AUC-ROC",   f"{roc_auc_score(y_test, y_proba):.3f}")

            st.markdown("---")

            perf_col1, perf_col2 = st.columns(2)

            # --- Matrice de confusion ---
            with perf_col1:
                st.subheader("🟦 Matrice de Confusion")
                cm = confusion_matrix(y_test, y_pred)
                fig_cm = px.imshow(
                    cm,
                    labels={"x": "Prédit", "y": "Réel", "color": "Nombre"},
                    x=["Rejeté", "Approuvé"],
                    y=["Rejeté", "Approuvé"],
                    text_auto=True,
                    color_continuous_scale="Blues",
                    title="Matrice de Confusion"
                )
                fig_cm.update_traces(textfont_size=18)
                st.plotly_chart(fig_cm, use_container_width=True)

            # --- Courbe ROC ---
            with perf_col2:
                st.subheader("📉 Courbe ROC")
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                auc_val = roc_auc_score(y_test, y_proba)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode="lines",
                    name=f"ROC (AUC = {auc_val:.3f})",
                    line=dict(color="#2ecc71", width=2)
                ))
                fig_roc.add_trace(go.Scatter(
                    x=[0, 1], y=[0, 1],
                    mode="lines",
                    name="Aléatoire",
                    line=dict(color="grey", dash="dash")
                ))
                fig_roc.update_layout(
                    title="Courbe ROC",
                    xaxis_title="Taux de Faux Positifs",
                    yaxis_title="Taux de Vrais Positifs",
                    legend=dict(x=0.6, y=0.1)
                )
                st.plotly_chart(fig_roc, use_container_width=True)

            st.markdown("---")

            feat_col1, feat_col2 = st.columns(2)

            # --- Feature Importance globale ---
            with feat_col1:
                st.subheader("🏆 Feature Importance Globale")
                st.warning("⚠️ Les importances sont basées sur les coefficients absolus du modèle — valables uniquement pour la Régression Logistique.")
                coefs    = model.coef_[0]
                feat_names = model.feature_names_in_
                importance_df = pd.DataFrame({
                    "Feature":    feat_names,
                    "Importance": np.abs(coefs)
                }).sort_values("Importance", ascending=True).tail(10)

                fig_fi = px.bar(
                    importance_df, x="Importance", y="Feature",
                    orientation="h",
                    title="Top 10 Features (|coefficient|)",
                    color="Importance",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_fi, use_container_width=True)

            # --- Distribution des probabilités ---
            with feat_col2:
                st.subheader("📊 Distribution des Probabilités")
                st.info("💡 Un bon modèle sépare nettement les deux distributions (rouge vs vert).")
                proba_df = pd.DataFrame({
                    "Probabilité": y_proba,
                    "Réalité": ["Approuvé" if v == 1 else "Rejeté" for v in y_test]
                })
                fig_proba = px.histogram(
                    proba_df, x="Probabilité", color="Réalité",
                    nbins=30, barmode="overlay",
                    title="Distribution des probabilités prédites",
                    color_discrete_map={"Approuvé": "#2ecc71", "Rejeté": "#e74c3c"},
                    opacity=0.7
                )
                fig_proba.add_vline(x=0.5, line_dash="dash", line_color="black",
                                    annotation_text="Seuil = 0.5")
                st.plotly_chart(fig_proba, use_container_width=True)

            # --- Expanders avec explications ---
            st.markdown("---")
            with st.expander("ℹ️ Que signifient ces métriques ?"):
                st.markdown("""
- **Accuracy** : % de prédictions correctes (approuvées + rejetées)
- **Precision** : Parmi les prêts prédits approuvés, % qui l'étaient vraiment
- **Recall** : Parmi les vrais prêts approuvés, % correctement détectés
- **F1-Score** : Moyenne harmonique de Precision et Recall (équilibre)
- **AUC-ROC** : Aire sous la courbe ROC (0.5 = aléatoire, 1.0 = parfait)
                """)

            with st.expander("ℹ️ Comment lire la matrice de confusion ?"):
                tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
                st.markdown(f"""
- **Vrais Négatifs (TN) = {tn}** : Correctement rejetés ✅
- **Faux Positifs (FP) = {fp}** : Faussement approuvés ⚠️ (risque financier)
- **Faux Négatifs (FN) = {fn}** : Faussement rejetés ⚠️ (opportunités manquées)
- **Vrais Positifs (TP) = {tp}** : Correctement approuvés ✅
                """)

            with st.expander("ℹ️ Comment interpréter la courbe ROC ?"):
                auc_interp = roc_auc_score(y_test, y_proba)
                qualite = "très bon" if auc_interp >= 0.8 else "bon"
                st.markdown(f"""
Plus la courbe s'éloigne de la diagonale, meilleur est le modèle.
Un AUC de **{auc_interp:.2f}** indique un **{qualite}** modèle.

- AUC = 0.5 → modèle aléatoire (inutile)
- AUC = 0.7–0.8 → bon modèle
- AUC > 0.8 → très bon modèle
- AUC = 1.0 → modèle parfait (souvent signe de surapprentissage)
                """)
                st.info("💡 Ces variables ont le plus d'impact sur les prédictions. "
                        "L'historique de crédit et les ratios financiers sont typiquement les plus importants.")

else:
    st.error("❌ Impossible de charger les données. Vérifiez que le fichier existe.")

# ============================================================================
# Section "À propos"
# ============================================================================
st.markdown("---")
with st.expander("📖 À propos de cette application"):
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown("### 🎯 Le projet")
        st.markdown("""
Application de prédiction d'approbation de prêt bancaire développée dans le cadre d'un projet ML.
Elle permet d'explorer les données, de faire des prédictions et d'évaluer les performances du modèle.
        """)
    with col_b:
        st.markdown("### ⚖️ Méthodologie & Éthique")
        st.markdown("""
- **RGPD** : Aucune donnée personnelle n'est stockée
- **Transparence** : Les décisions sont expliquées via les features influentes
- **Biais** : Le modèle a été audité pour détecter les biais (genre, zone géographique)
- **Données** : Dataset public Kaggle — Loan Prediction Dataset
        """)
    with col_c:
        st.markdown("### 📬 Contact")
        st.markdown("""
- **École** : NEXA Digital School
- **Promotion** : M2 — 2025/2026
- **Modèle** : Régression Logistique (sklearn)
        """)

# ============================================================================
# Footer avec version et date
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: grey; font-size: 0.85em;'>
        🏦 <b>Loan Approval App</b> — v1.0.0 &nbsp;|&nbsp;
        Dernière mise à jour : Mars 2026 &nbsp;|&nbsp;
        Projet Machine Learning — NEXA Digital School
    </div>
    """,
    unsafe_allow_html=True,
)
