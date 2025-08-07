import streamlit as st
import xgboost as xgb
import plotly.express as px
import shap
import pandas as pd
import numpy as np
import plotly.graph_objects as go 
import joblib
import matplotlib.pyplot as plt
from PIL import Image


#--------------------------------
## Logo dans la sidebar
#--------------------------------
@st.cache_data
def load_logo():
    try:
        image = Image.open('logo.png')
        return image
    except FileNotFoundError:
        return None

logo = load_logo()
if logo:
    st.sidebar.image(logo, use_container_width=True)

# Charger le modèle et les colonnes encodées
# ----------------------------
@st.cache_resource
def load_model():
    model = joblib.load("model.joblib")
    columns = joblib.load("columns.joblib")  # Colonnes après get_dummies
    return model, columns

model, model_columns = load_model()


# ----------------------------------
# Générer la feature importance globale
# -----------------------------------

@st.cache_resource
def compute_global_shap_values(_model, _data_sample):
    explainer = shap.Explainer(_model)
    shap_values = explainer(_data_sample)
    return explainer, shap_values


# ----------------------------
# Fonction : Cadrant (compteur visuel)
# ----------------------------
def plot_gauge_chart(score):
    fig = go.Figure(go.Indicator(
        domain={'x': [0, 1], 'y': [0, 1]},
        value=score,
        mode="gauge+number",
        title={'text': "Probabilité de défaut", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 1]},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 0.3], 'color': 'green'},
                {'range': [0.3, 0.7], 'color': 'yellow'},
                {'range': [0.7, 1], 'color': 'red'}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': score
            }}))

    fig.update_layout(font={'color': "black", 'family': "Arial"},
                      paper_bgcolor='rgba(255,255,255,1)',
                      plot_bgcolor='rgba(255,255,255,1)',
                      margin=dict(l=20, r=20, t=80, b=20))
    
    return fig


# ----------------------------
# Interface utilisateur
# ----------------------------
st.title(" Dashboard de Prédiction")
st.write("Entrez un identifiant client pour estimer la probabilité de défaut.")

# Charger les données clients
df_clients = pd.read_csv("./data/data_cleaned.csv")
client_ids = df_clients["SK_ID_CURR"].tolist()
client_id = st.selectbox("Sélectionnez un client :", client_ids)

# On prend un échantillon de données pour ne pas surcharger SHAP
global_sample = df_clients.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore")
global_sample_encoded = pd.get_dummies(global_sample)
global_sample_encoded = global_sample_encoded.reindex(columns=model_columns, fill_value=0)

explainer, global_shap_values = compute_global_shap_values(_model=model, _data_sample =global_sample_encoded.sample(100))

# ----------------------------
# Préparer les données du client
# ----------------------------
client_data = df_clients[df_clients["SK_ID_CURR"] == client_id]
X_client = client_data.drop(columns=["SK_ID_CURR", "TARGET"], errors="ignore")

# Affichage des features du client
st.subheader(" Informations du client sélectionné")
st.dataframe(client_data.drop(columns=["TARGET"], errors="ignore"))

# Encodage
X_client_encoded = pd.get_dummies(X_client)

# Aligner avec les colonnes du modèle
X_client_encoded = X_client_encoded.reindex(columns=model_columns, fill_value=0)

# Optionnel : afficher les features utilisées
with st.expander(" Voir les features utilisées pour la prédiction"):
    st.dataframe(X_client_encoded.T.rename(columns={0: "Valeur"}))

# ----------------------------
# Prédiction
# ----------------------------
proba = model.predict_proba(X_client_encoded)[0, 1]
st.subheader(" Résultat de la prédiction :")
st.metric(label="Probabilité de défaut", value=f"{proba:.2%}")
fig = plot_gauge_chart(proba)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Interprétation locale avec SHAP
# ----------------------------
with st.expander(" Interprétation locale des features (SHAP)"):
    try:
        # Créer l'explainer SHAP
        explainer = shap.Explainer(model)

        # Calculer les valeurs SHAP du client
        shap_values = explainer(X_client_encoded)

        # Afficher le graphe waterfall

        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(bbox_inches='tight')
    except Exception as e:
        st.error(f"Erreur lors de l'explication SHAP : {e}")

#------------------------
# Visualisation des caractéristiques du client par rapport à la population
# --------------------------

st.subheader(" Comparaison des caractéristiques client vs population")

# Liste des colonnes disponibles (on exclut ID et TARGET s'ils sont là)
available_features = [col for col in df_clients.columns if col not in ["SK_ID_CURR", "TARGET"]]

# Choix de la feature à visualiser
selected_feature = st.selectbox("Sélectionnez une feature à comparer :", available_features)

# Détection du type
if pd.api.types.is_numeric_dtype(df_clients[selected_feature]):
    # Histogramme pour les valeurs numériques
    fig = px.histogram(df_clients, x=selected_feature, nbins=30,
                       title=f"Distribution de {selected_feature} pour l'ensemble des clients",
                       opacity=0.7, marginal="box")

    fig.add_vline(x=float(client_data[selected_feature].values[0]), line_dash="dash", line_color="red",
                  annotation_text="Client", annotation_position="top right")

else:
    # Bar chart pour les valeurs catégorielles
    counts = df_clients[selected_feature].value_counts().reset_index()
    counts.columns = [selected_feature, 'count']
    fig = px.bar(counts, x=selected_feature, y='count',
                 title=f"Répartition des catégories pour {selected_feature}")

    client_val = client_data[selected_feature].values[0]
    fig.add_trace(go.Bar(x=[client_val], y=[df_clients[selected_feature].value_counts()[client_val]],
                         marker_color='red', name='Client'))

st.plotly_chart(fig, use_container_width=True)

#------------------------------------------
# Analyse Bi-variée-------------------
#-------------------------------------

st.subheader("Analyse bi-variée")

feature_x = st.selectbox("Sélectionnez la 1ère feature :", df_clients.columns)
feature_y = st.selectbox("Sélectionnez la 2ème feature :", df_clients.columns)

try:
    if pd.api.types.is_numeric_dtype(df_clients[feature_x]) and pd.api.types.is_numeric_dtype(df_clients[feature_y]):
        fig = px.scatter(df_clients, x=feature_x, y=feature_y,
                         color="TARGET", title=f"{feature_x} vs {feature_y}",
                         labels={"TARGET": "Défaut"})
    
    elif pd.api.types.is_categorical_dtype(df_clients[feature_x]) or df_clients[feature_x].dtype == object:
        fig = px.box(df_clients, x=feature_x, y=feature_y,
                     color="TARGET", title=f"{feature_x} vs {feature_y} par défaut")

    elif pd.api.types.is_categorical_dtype(df_clients[feature_y]) or df_clients[feature_y].dtype == object:
        fig = px.box(df_clients, x=feature_y, y=feature_x,
                     color="TARGET", title=f"{feature_y} vs {feature_x} par défaut")
    
    else:
        st.warning("Ce type de combinaison n'est pas encore géré.")
        fig = None

    if fig:
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Erreur lors de l'affichage du graphique : {e}")

#-----------------------------
# Calcul de la features importance locale 
#----------------------------
st.subheader("Comparaison : SHAP Local vs Global")

# Valeur du client sélectionné
try:
    shap_client = explainer(X_client_encoded)

    # Affichage côte à côte
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Feature Importance Locale (Client)**")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_client[0],ax=ax, show=False)
        st.pyplot(fig)

    with col2:
        st.markdown("**Feature Importance Globale**")
        fig2, ax2 = plt.subplots()
        shap.plots.bar(global_shap_values, max_display=10,ax=ax2, show=False)
        st.pyplot(fig2)

except Exception as e:
    st.error(f"Erreur lors du calcul ou affichage SHAP : {e}")
