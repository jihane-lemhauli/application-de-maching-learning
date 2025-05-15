import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Titre
st.title("Application de Machine Learning - Prédiction Client Bancaire")

# Chargement des données
uploaded_file = st.file_uploader("Choisissez le fichier CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Aperçu des données")
    st.dataframe(df.head())

    # Encodage
    df_encoded = df.copy()
    label_encoders = {}
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    for column in categorical_cols:
        le = LabelEncoder()
        df_encoded[column] = le.fit_transform(df_encoded[column])
        label_encoders[column] = le

    # Définir la colonne cible
    target_col = "deposit" if "deposit" in df.columns else "y"
    X = df_encoded.drop(target_col, axis=1)
    y = df_encoded[target_col]

    # Normalisation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Choix du modèle
    model_name = st.selectbox("Choisir le modèle :", ["Logistic Regression", "KNN", "Random Forest"])

    if model_name == "Logistic Regression":
        model = LogisticRegression()
    elif model_name == "KNN":
        model = KNeighborsClassifier()
    else:
        model = RandomForestClassifier()

    # Entraînement
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("Résultats de Prédiction")
    st.text(classification_report(y_test, y_pred))

    # Matrice de confusion
    st.subheader("Matrice de Confusion")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    st.pyplot(fig)

    # Courbe ROC ( binaire)
    st.subheader("Courbe ROC")
    if len(np.unique(y)) == 2:
        y_score = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        fig2, ax2 = plt.subplots()
        ax2.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.2f' % roc_auc)
        ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('Receiver Operating Characteristic')
        ax2.legend(loc="lower right")
        st.pyplot(fig2)

    # Prédiction personnalisée
    st.subheader("Faire une prédiction personnalisée")
    sample = []

    for col in X.columns:
        if col in categorical_cols:
            # afficher la liste des choix texte pour cette colonne
            options = label_encoders[col].classes_
            choice = st.selectbox(f"{col}", options)
            # convertir le choix en code numérique
            val = label_encoders[col].transform([choice])[0]
            sample.append(val)
        else:
            # pour les colonnes numériques, garder un input numérique avec valeur moyenne par défaut
            val = st.number_input(f"{col}", value=float(df[col].mean()))
            sample.append(val)

    if st.button("Prédire"):
        sample_np = np.array(sample).reshape(1, -1)
        sample_np = scaler.transform(sample_np)
        prediction = model.predict(sample_np)
        # retrouver le label texte de la prédiction si encodé
        if target_col in label_encoders:
            label = label_encoders[target_col].inverse_transform(prediction)[0]
        else:
            label = prediction[0]
        st.success(f"Résultat de prédiction : {label}")
