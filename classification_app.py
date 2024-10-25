import gradio as gr
from tensorflow.keras.models import load_model  # type: ignore
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
from tensorflow.keras.layers import DepthwiseConv2D  # type: ignore

# Définir la couche sans le paramètre 'groups'
def custom_depthwise_conv2d(*args, **kwargs):
    if 'groups' in kwargs:
        del kwargs['groups']  # Retirer 'groups'
    return DepthwiseConv2D(*args, **kwargs)

# Charger le modèle
model = load_model("models/keras_model.h5", custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d}, compile=False)

# Charger les étiquettes
with open("labels.txt", "r") as file:
    class_names = file.readlines()

# Fonction pour prédire la classe d'une image
def predict(image):
    # Redimensionner l'image à 224x224
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    
    # Convertir l'image en tableau numpy
    image_array = np.asarray(image)
    
    # Normaliser l'image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    
    # Créer le tableau de données pour le modèle
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # Prédire avec le modèle
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip()  # Supprimer les espaces supplémentaires
    confidence_score = prediction[0][index]
    
    return class_name, confidence_score

# Créer l'interface Gradio
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Téléchargez une image"),
    outputs=[
        gr.Label(label="Classe Prédite"),
        gr.Number(label="Score de Confiance")
    ],
    title="Medical Assistant",
    description="Téléchargez une image dentaire et notre application prédira sa classe."
)

# Lancer l'interface
iface.launch(share=True, debug=True, show_api=False)