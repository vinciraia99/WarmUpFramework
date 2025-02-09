import matplotlib
import pandas as pd
import numpy as np
import tensorflow as tf
import logging
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Configura Matplotlib per utilizzare un backend adatto
matplotlib.use('TkAgg')

# Configura il logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_and_prepare_data(file_paths):
    """
    Carica e prepara i dati:
    - Combina i dati da più file.
    - Esegue il bilanciamento delle classi per gestire dataset sbilanciati.
    """
    logger.info("Caricamento e preparazione dei dati...")
    all_data = []
    for file_path in file_paths:
        logger.info(f"Caricamento del file: {file_path}")
        data = pd.read_csv(file_path)  # Legge il file CSV
        all_data.append(data)

    # Combina i dati in un unico DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)
    logger.info("Dati combinati con successo.")

    # Separazione delle classi
    stable_blocks = combined_data[combined_data['is_stable'] == True]
    unstable_blocks = combined_data[combined_data['is_stable'] == False]

    # Bilanciamento delle classi basato sul numero minimo di campioni
    min_samples = min(len(stable_blocks), len(unstable_blocks))
    stable_blocks_balanced = resample(stable_blocks, n_samples=min_samples, random_state=42)
    unstable_blocks_balanced = resample(unstable_blocks, n_samples=min_samples, random_state=42)

    # Combina i dati bilanciati
    balanced_data = pd.concat([stable_blocks_balanced, unstable_blocks_balanced])
    logger.info("Bilanciamento completato.")
    return balanced_data


def preprocess_data(data):
    """
    Prepara i dati per l'addestramento:
    - Estrae le feature (X) e la variabile target (y).
    - Suddivide i dati in training e test set.
    - Ridimensiona i dati per adattarli al modello FCN.
    """
    logger.info("Preprocessing dei dati...")
    # Selezione delle feature e della variabile target
    X = data[['mean', 'std_dev', 'min', 'max', 'median']].values
    y = data['is_stable'].astype(int).values

    # Suddivisione in training e test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Adatta i dati per l'input della rete convoluzionale
    X_train = np.expand_dims(X_train, axis=2)  # Aggiunge una dimensione per la convoluzione 1D
    X_test = np.expand_dims(X_test, axis=2)
    logger.info("Preprocessing completato.")
    return X_train, X_test, y_train, y_test


def build_model(input_shape):
    """
    Costruisce una Fully Convolutional Network (FCN):
    - Utilizza strati convolutivi per estrarre pattern dalle feature numeriche.
    - Include uno strato GlobalAveragePooling per ridurre dimensionalità.
    - Output: classificazione binaria tramite sigmoid.
    """
    logger.info("Costruzione del modello FCN...")
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu', padding='same', input_shape=input_shape),
        tf.keras.layers.Conv1D(filters=256, kernel_size=2, activation='relu', padding='same'),
        tf.keras.layers.Conv1D(filters=128, kernel_size=2, activation='relu', padding='same'),
        tf.keras.layers.GlobalAveragePooling1D(),  # Riduce la dimensionalità mantenendo informazioni globali
        tf.keras.layers.Dense(1, activation='sigmoid')  # Classificazione binaria
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    logger.info("Modello costruito con successo.")
    return model


def evaluate_model(model, X_test, y_test):
    """
    Valuta il modello:
    - Genera un report di classificazione con precision, recall e f1-score.
    - Crea una matrice di confusione e la visualizza.
    """
    logger.info("Valutazione del modello...")
    y_pred = (model.predict(X_test) > 0.5).astype(int)  # Predizioni binarie

    # Genera la matrice di confusione
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Estrai i valori dalla matrice di confusione
    TN, FP, FN, TP = conf_matrix.ravel()

    # Crea un DataFrame per mostrare i risultati
    results_df = pd.DataFrame({
        'Classe': ['Stable', 'Unstable'],
        'Veri Positivi': [TP, TN],
        'Falsi Positivi': [FP, FN],
        'Falsi Negativi': [FN, FP],
        'Veri Negativi': [TN, TP]
    })

    # Stampa il DataFrame
    print(results_df)

    # Salva il DataFrame in un file CSV
    results_df.to_csv("confusion_matrix_results.csv", index=False)

    # Visualizza la matrice di confusione
    plt.figure(figsize=(6, 4))
    sns.heatmap(pd.DataFrame(conf_matrix, index=['Actual Unstable', 'Actual Stable'],
                             columns=['Predicted Unstable', 'Predicted Stable']),
                annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    logger.info("Valutazione completata.")
    return results_df, conf_matrix


def save_metrics_as_image(report, conf_matrix, filename="metrics_output.png"):
    """
    Salva le metriche di classificazione come immagine:
    - Report di classificazione.
    - Matrice di confusione.
    """
    logger.info(f"Salvataggio delle metriche come immagine: {filename}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Visualizza il report come heatmap
    report_df = pd.DataFrame(report).transpose()
    sns.heatmap(report_df.iloc[:-1, :].T, annot=True, fmt=".2f", cmap="Blues", cbar=False, ax=ax[0])
    ax[0].set_title("Classification Report")

    # Visualizza la matrice di confusione
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[1])
    ax[1].set_title("Confusion Matrix")
    ax[1].set_xlabel("Predicted")
    ax[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(filename)
    logger.info(f"Metriche salvate con successo in {filename}.")


# **FLUSSO DI LAVORO**
file_paths = ["../Dataset misurazioni/dataset__Ticket Reserve_API.csv", "../Dataset misurazioni/dataset_Captcha.csv","../Dataset misurazioni/dataset_Train.csv","../Dataset misurazioni/datasetOrderVoucher.csv"]

# 1. Caricamento e preparazione dei dati
logger.info("Inizio del flusso di lavoro...")
data = load_and_prepare_data(file_paths)

# 2. Preprocessing dei dati
X_train, X_test, y_train, y_test = preprocess_data(data)

# 3. Costruzione e addestramento del modello
model = build_model(input_shape=(X_train.shape[1], 1))
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)

# 4. Valutazione del modello
loss, accuracy = model.evaluate(X_test, y_test)
logger.info(f"Test Loss: {loss}")
logger.info(f"Test Accuracy: {accuracy}")
report, conf_matrix = evaluate_model(model, X_test, y_test)

# 5. Salvataggio delle metriche e del modello
save_metrics_as_image(report, conf_matrix)
model.save("fcn_binary_classification_model.h2", save_format="keras")

# 6. Plot delle curve di addestramento
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()
plt.show()
logger.info("Flusso di lavoro completato.")
