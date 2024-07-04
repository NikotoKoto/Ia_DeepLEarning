import openai
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from typing import List
from PIL import Image, ImageOps
import io
import numpy as np
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import requests
import logging
import os

app = FastAPI(
    title="Image Classification API",
    description="An API for training and classifying images using Machine Learning models.",
    version="1.0.0",
)

TAILLE_IMAGE = (256, 256)  # Taille standardisée des images

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Charger un modèle si existant, sinon en créer un nouveau
try:
    model = tf.keras.models.load_model("models/model.h5")
    label_encoder = joblib.load("models/label_encoder.joblib")
except:
    model = None
    label_encoder = LabelEncoder()

def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = ImageOps.fit(image, TAILLE_IMAGE, Image.LANCZOS)  # Redimensionner avec padding 
    return np.array(image) / 255.0  

@app.get("/", summary="Message de bienvenue", tags=["General"])
async def read_root():
    """
    Bienvenue à l'API de Classification d'Images.
    Utilisez cette API pour entraîner des modèles de classification d'images et classer des images.
    """
    return {"message": "Bienvenue à l'API de Classification d'Images"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(os.path.join(os.path.dirname(__file__), 'favicon.ico'))

@app.post("/train", summary="Entraîner un modèle", tags=["Training"])
async def train_model(files: List[UploadFile] = File(...), labels: str = Form(...)):
    """
    Entraîner un modèle avec les images et les étiquettes fournies.
    - **files**: Liste des fichiers d'images à utiliser pour l'entraînement.
    - **labels**: Étiquettes des images, séparées par des virgules.
    """
    try:
        labels = labels.split(',')
        if len(files) != len(labels):
            return JSONResponse(status_code=400, content={"message": "Le nombre d'étiquettes doit correspondre au nombre d'images téléchargées."})

        # Encoder les étiquettes
        label_encoder.fit(labels)
        encoded_labels = label_encoder.transform(labels)

        data = []
        for file in files:
            image = Image.open(io.BytesIO(await file.read()))
            processed_image = preprocess_image(image)
            data.append(processed_image)
        
        X = np.array(data)
        y = np.array(encoded_labels)
        
        # Équilibrer les données
        ros = RandomOverSampler(random_state=42)
        X_resampled, y_resampled = ros.fit_resample(X.reshape(len(X), -1), y)
        X_resampled = X_resampled.reshape(-1, 256, 256, 3)
        
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
        
        # Augmentation de données
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Utiliser MobileNetV2 comme modèle de base
        base_model = MobileNetV2(input_shape=(256, 256, 3), include_top=False, weights='imagenet')
        base_model.trainable = False  # Geler les couches du modèle de base

        model = Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.5),
            Dense(256, activation='relu'),
            Dropout(0.5),
            Dense(len(label_encoder.classes_), activation='softmax')
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        # Entraîner le modèle avec l'augmentation de données
        history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_test, y_test))
        
        loss, accuracy = model.evaluate(X_test, y_test)
        
        model.save("models/model.h5")
        joblib.dump(label_encoder, "models/label_encoder.joblib")

        return {"message": "Modèle entraîné avec succès", "accuracy": accuracy}
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return JSONResponse(status_code=500, content={"message": f"Une erreur s'est produite: {e}"})

@app.post("/predict", summary="Prédire une image", tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)):
    """
    Faire une prédiction avec le dernier modèle sauvegardé.
    - **file**: Fichier d'image à prédire.
    """
    try:
        model = tf.keras.models.load_model("models/model.h5")
        label_encoder = joblib.load("models/label_encoder.joblib")
        
        image = Image.open(io.BytesIO(await file.read()))
        processed_image = preprocess_image(image).reshape(1, 256, 256, 3)
        classification = np.argmax(model.predict(processed_image), axis=1)[0]
        label = label_encoder.inverse_transform([classification])[0]
        return {"filename": file.filename, "classification": label}
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return JSONResponse(status_code=500, content={"message": f"Une erreur s'est produite: {e}"})

@app.post("/classify_images", summary="Classer deux images", tags=["Prediction"])
async def classify_images(files: List[UploadFile] = File(...)):
    """
    Classer deux images téléchargées.
    - **files**: Liste de deux fichiers d'images à classer.
    """
    if len(files) != 2:
        return JSONResponse(status_code=400, content={"message": "Veuillez télécharger exactement deux images."})

    results = []
    try:
        model = tf.keras.models.load_model("models/model.h5")
        label_encoder = joblib.load("models/label_encoder.joblib")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return JSONResponse(status_code=500, content={"message": f"Erreur de chargement du modèle: {e}"})

    for file in files:
        try:
            image = Image.open(io.BytesIO(await file.read()))
            processed_image = preprocess_image(image).reshape(1, 256, 256, 3)
            classification = np.argmax(model.predict(processed_image), axis=1)[0]
            label = label_encoder.inverse_transform([classification])[0]
            results.append({"filename": file.filename, "classification": label})
        except Exception as e:
            logger.error(f"Error processing image {file.filename}: {e}")
            return JSONResponse(status_code=500, content={"message": f"Erreur lors du traitement de l'image {file.filename}: {e}"})

    return {"results": results}

@app.get("/model", summary="Obtenir des informations sur un modèle", tags=["Model Information"])
async def get_model_info(api_choice: str):
    """
    Appeler l'API de OpenAI ou HuggingFace pour obtenir des informations sur un modèle.
    - **api_choice**: Choix de l'API à appeler ('openai' ou 'huggingface').
    """
    if api_choice not in ["openai", "huggingface"]:
        raise HTTPException(status_code=400, detail="Choix de l'API invalide. Utilisez 'openai' ou 'huggingface'.")

    try:
        if api_choice == "openai":
            # Appel à l'API OpenAI
            openai.api_key = os.getenv("OPENAI_API_KEY")
            response = openai.Engine.list()  
            return response
        else:
            # Appel à l'API HuggingFace
            response = requests.get("https://huggingface.co/api/models")
            if response.status_code == 200:
                return response.json()
            else:
                raise HTTPException(status_code=response.status_code, detail="Erreur lors de l'appel à l'API HuggingFace")
    except Exception as e:
        logger.error(f"Error calling {api_choice} API: {e}")
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'appel à l'API {api_choice}: {e}")

