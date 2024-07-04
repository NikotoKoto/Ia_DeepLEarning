import streamlit as st
import requests

# URL de l'API
API_URL_TRAIN = "http://127.0.0.1:8000/train"
API_URL_PREDICT = "http://127.0.0.1:8000/predict"
API_URL_CLASSIFY = "http://127.0.0.1:8000/classify_images"
API_URL_MODEL = "http://127.0.0.1:8000/model"

st.title("Application de Classification d'Images")

# Section d'entraînement du modèle
st.header("Entraîner le Modèle")
with st.form(key='train_form'):
    class1_name = st.text_input("Entrez le nom de la première classe")
    class2_name = st.text_input("Entrez le nom de la deuxième classe")
    
    class1_files = st.file_uploader(f"Choisissez des images pour la classe: {class1_name}", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="class1")
    class2_files = st.file_uploader(f"Choisissez des images pour la classe: {class2_name}", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="class2")
    
    train_button = st.form_submit_button(label='Entraîner le Modèle')

    if train_button:
        if class1_files and class2_files and class1_name and class2_name:
            files = []
            labels = []
            
            for file in class1_files:
                files.append(("files", (file.name, file, file.type)))
                labels.append(class1_name)
            
            for file in class2_files:
                files.append(("files", (file.name, file, file.type)))
                labels.append(class2_name)
            
            response = requests.post(API_URL_TRAIN, files=files, data={"labels": ",".join(labels)})
            if response.status_code == 200:
                try:
                    result = response.json()
                    st.success(f"Modèle entraîné avec succès avec une précision de: {result['accuracy']:.2f}")
                except ValueError:
                    st.error("Erreur lors de l'analyse de la réponse JSON")
            else:
                try:
                    error_message = response.json().get('message', 'Erreur inconnue')
                except ValueError:
                    error_message = response.text
                st.error(f"Erreur lors de l'entraînement du modèle: {error_message}")
        else:
            st.error("Veuillez fournir des noms pour les deux classes et télécharger des images pour chaque classe.")

# Section pour classer des images
st.header("Classer des Images")
uploaded_files = st.file_uploader("Choisissez deux images à classer", type=["jpg", "png", "jpeg"], accept_multiple_files=True, key="classify")

if uploaded_files and len(uploaded_files) == 2:
    files = [("files", (file.name, file, file.type)) for file in uploaded_files]
    response = requests.post(API_URL_CLASSIFY, files=files)
    if response.status_code == 200:
        try:
            results = response.json()["results"]
            for result in results:
                st.write(f"Image `{result['filename']}` classée comme `{result['classification']}`")
        except ValueError:
            st.error("Erreur lors de l'analyse de la réponse JSON")
    else:
        try:
            error_message = response.json().get('message', 'Erreur inconnue')
        except ValueError:
            error_message = response.text
        st.error(f"Erreur lors de la classification des images: {error_message}")
else:
    if uploaded_files:
        st.error("Veuillez télécharger exactement deux images.")

# Section pour prédire une seule image
st.header("Prédire une Image")
uploaded_file = st.file_uploader("Choisissez une image à prédire", type=["jpg", "png", "jpeg"], key="predict")

if uploaded_file is not None:
    response = requests.post(API_URL_PREDICT, files={"file": (uploaded_file.name, uploaded_file, uploaded_file.type)})
    if response.status_code == 200:
        try:
            result = response.json()
            st.write(f"Image `{result['filename']}` classée comme `{result['classification']}`")
        except ValueError:
            st.error("Erreur lors de l'analyse de la réponse JSON")
    else:
        try:
            error_message = response.json().get('message', 'Erreur inconnue')
        except ValueError:
            error_message = response.text
        st.error(f"Erreur lors de la prédiction de l'image: {error_message}")

# Section pour appeler l'API OpenAI ou HuggingFace
st.header("Obtenir des Informations sur un Modèle")
api_choice = st.selectbox("Choisissez l'API", ["openai", "huggingface"])
if st.button("Obtenir des Informations sur le Modèle"):
    try:
        response = requests.get(API_URL_MODEL, params={"api_choice": api_choice})
        if response.status_code == 200:
            try:
                st.json(response.json())
            except ValueError:
                st.error("Réponse de l'API invalide")
        else:
            try:
                error_message = response.json().get('detail', 'Erreur inconnue')
            except ValueError:
                error_message = response.text
            st.error(f"Erreur lors de l'appel à l'API {api_choice}: {error_message}")
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur de connexion: {e}")
