from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import logging
import joblib
import tensorflow as tf
import os
import random


# Modello dei dati di input
class FullPayload(BaseModel):
    responseCode: int
    responseMessage: str
    responseTime: float
    connectTime: float
    latency: float
    bytes: int
    sentBytes: int
    success: bool
    responseData: str
    threadName: str
    sampleLabel: str


# Percorsi dei modelli
ROCKET_MODEL_PATH = "./Models/ridge_classifier_model.pkl"
KERNELS_PATH = "./Models/rocket_kernels.pkl"
FCN_MODEL_PATH = "./Models/fcn_binary_classification_model.h5"

# Buffer per i dati
BUFFER_SIZE = 50
buffer = []


# Funzioni ROCKET
def generate_random_kernels(input_length, num_kernels):
    """
    Genera kernel convoluzionali casuali.
    """
    kernels = []
    for _ in range(num_kernels):
        length = random.randint(2, input_length)
        weights = np.random.randn(length)
        bias = np.random.randn()
        kernels.append((weights, bias))
    return kernels


def apply_kernel(data, kernel):
    weights, bias = kernel
    convolved = np.array([np.convolve(x, weights, mode='valid') + bias for x in data])
    max_vals = np.max(convolved, axis=1)
    positive_props = np.mean(convolved > 0, axis=1)
    return max_vals, positive_props


def transform_with_rocket(X, kernels):
    features = []
    for kernel in kernels:
        max_vals, positive_props = apply_kernel(X, kernel)
        features.append(max_vals)
        features.append(positive_props)
    return np.column_stack(features)


# Configura il logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ML_API")

# Caricamento dei modelli e dei kernel
logger.info("Caricamento del modello Ridge")
rocket_model = joblib.load(ROCKET_MODEL_PATH)
try:
    kernels = joblib.load(KERNELS_PATH)
except FileNotFoundError:
    logger.warning("File dei kernel non trovato. Generazione di nuovi kernel.")
    kernels = generate_random_kernels(input_length=5, num_kernels=50)
logger.info("Caricamento modello FCN")
fcn_model = tf.keras.models.load_model(FCN_MODEL_PATH)
logger.info("Modelli e kernel pronti")

# Inizializza l'app FastAPI
app = FastAPI()


def process_buffer_for_model(model, kernels=None):
    global buffer
    if len(buffer) < BUFFER_SIZE:
        logger.info(f"Buffer non pieno: {len(buffer)}/{BUFFER_SIZE}")
        return False

    # Aggrega i dati nel buffer
    latencies = np.array(buffer)
    mean = np.mean(latencies[:, 0])
    std_dev = np.std(latencies[:, 0])
    min_val = np.min(latencies[:, 0])
    max_val = np.max(latencies[:, 0])
    median = np.median(latencies[:, 0])
    X = np.array([[mean, std_dev, min_val, max_val, median]])

    if kernels:
        X_transformed = transform_with_rocket(X, kernels)
        prediction = bool(model.predict(X_transformed)[0])
    else:
        X = np.expand_dims(X, axis=2)
        prediction = bool(model.predict(X)[0][0] > 0.5)

    if prediction:
        logger.info("Blocco stabile rilevato. Svuotamento del buffer.")
        buffer.clear()
        return True
    else:
        logger.info("Blocco non stabile. Continuazione con nuovo blocco.")
        buffer.pop(0)
        return False


@app.post("/rocket")
def rocket_endpoint(payload: FullPayload):
    global buffer
    logger.info(f"Ricevuto input: {payload}")
    buffer.append([payload.latency])
    if len(buffer) > BUFFER_SIZE:
        buffer.pop(0)

    result = process_buffer_for_model(rocket_model, kernels=kernels)
    return "true" if result else "false"


@app.post("/fcn")
def fcn_endpoint(payload: FullPayload):
    global buffer
    logger.info(f"Ricevuto input: {payload}")
    buffer.append([payload.latency])
    if len(buffer) > BUFFER_SIZE:
        buffer.pop(0)

    result = process_buffer_for_model(fcn_model)
    return "true" if result else "false"


if __name__ == "__main__":
    import uvicorn

    port = 5000
    logger.info(f"Avvio del server sulla porta {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
