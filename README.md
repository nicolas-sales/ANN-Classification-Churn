## Customer Churn Prediction — App Streamlit

Application Streamlit pour prédire la probabilité de churn (départ client) à partir de caractéristiques client.
Le modèle est un réseau Keras (model.h5) et s’appuie sur des artefacts de prétraitement scikit-learn sérialisés :

onehot_encoder_geo.pkl : OneHotEncoder pour Geography (avec handle_unknown="ignore").

label_encoder_gender.pkl : LabelEncoder (ou mapping 0/1) pour Gender.

scaler.pkl : StandardScaler (ou équivalent) entraîné sur les mêmes colonnes et dans le même ordre que lors du training.

# Entrées utilisateur (UI)

Geography : sélection (catégories vues par l’OHE entraîné).

Gender : sélection (classes vues par le LabelEncoder).

CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary.

# Pipeline d’inférence 

Lecture des valeurs saisies dans l’interface Streamlit.

Encodage :

Gender → 0/1 via label_encoder_gender.transform([gender]).

Geography → one-hot via onehot_encoder_geo.transform([[geography]]), puis DataFrame avec get_feature_names_out.

Assemblage d’une unique ligne de features.

Alignement des colonnes sur celles attendues par le scaler (mêmes noms et ordre) :

si disponible : scaler.feature_names_in_ comme source de vérité ;

sinon, lister manuellement l’ordre attendu (celui du training).

Mise à l’échelle : scaler.transform(...).

Prédiction : model.predict(...) → probabilité de churn (sigmoïde).

Affichage de la proba + verdict (seuil par défaut 0.5).

# Lancement et ouverture de l'appplication
streamlit run app.py
Ouverture ensuite avec le lien local affiché
