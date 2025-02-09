import logging
import random
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import RidgeClassifierCV
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import seaborn as sns

# Configura il backend di Matplotlib
matplotlib.use('TkAgg')

# Configura il logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_balance_data(file_paths):
    """
    Carica i dati da più file, combina i dataset e bilancia le classi.
    """
    logging.info("Caricamento dei file e bilanciamento dei dati...")
    all_data = []

    # Caricamento dei dati
    for file_path in file_paths:
        logging.info(f"Caricamento del file: {file_path}")
        data = pd.read_csv(file_path)
        all_data.append(data)

    # Combina i dati in un unico DataFrame
    combined_data = pd.concat(all_data, ignore_index=True)
    logging.info("Dati combinati con successo.")

    # Prepara le feature (X) e le etichette (y)
    X = combined_data[['mean', 'std_dev', 'min', 'max', 'median']].values
    y = combined_data['is_stable'].astype(int).values

    # Bilanciamento delle classi
    ros = RandomOverSampler(random_state=42)
    X_balanced, y_balanced = ros.fit_resample(X, y)
    logging.info("Bilanciamento completato. Classi bilanciate: %s", dict(pd.Series(y_balanced).value_counts()))

    # Crea un DataFrame bilanciato
    balanced_data = pd.DataFrame(X_balanced, columns=['mean', 'std_dev', 'min', 'max', 'median'])
    balanced_data['is_stable'] = y_balanced
    return balanced_data

def generate_random_kernels(input_length, num_kernels):
    """
    Genera kernel convoluzionali casuali.
    """
    logging.info("Generazione di %d kernel casuali con lunghezza massima %d", num_kernels, input_length)
    kernels = []
    for _ in range(num_kernels):
        length = random.randint(2, input_length)
        weights = np.random.randn(length)
        bias = np.random.randn()
        kernels.append((weights, bias))
    return kernels

def apply_kernel(data, kernel):
    """
    Applica un kernel convoluzionale casuale ai dati.
    """
    weights, bias = kernel
    convolved = np.array([np.convolve(x, weights, mode='valid') + bias for x in data])
    max_vals = np.max(convolved, axis=1)
    positive_props = np.mean(convolved > 0, axis=1)
    return max_vals, positive_props

def transform_with_rocket(X, kernels):
    """
    Trasforma i dati usando ROCKET (Random Convolutional Kernel Transform).
    """
    logging.info("Trasformazione dei dati usando ROCKET con %d kernel", len(kernels))
    features = []
    for kernel in kernels:
        max_vals, positive_props = apply_kernel(X, kernel)
        features.append(max_vals)
        features.append(positive_props)
    return np.column_stack(features)

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

    logging.info("Valutazione completata.")
    return results_df, conf_matrix

def save_metrics_as_image(report, conf_matrix, filename="metrics_output.png"):
    """
    Salva le metriche di classificazione come immagine:
    - Report di classificazione.
    - Matrice di confusione.
    """
    logging.info(f"Salvataggio delle metriche come immagine: {filename}")
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
    logging.info(f"Metriche salvate con successo in {filename}.")

def plot_metrics(report, conf_matrix):
    """
    Crea e salva i grafici della matrice di confusione e delle metriche di classificazione.
    """
    logging.info("Creazione dei grafici delle metriche.")

    # Grafico della matrice di confusione
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Unstable", "Stable"], yticklabels=["Unstable", "Stable"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    logging.info("Matrice di confusione salvata come confusion_matrix.png")
    plt.show()

    # Grafico del report di classificazione
    report_df = pd.DataFrame(report).transpose()
    report_df[["precision", "recall", "f1-score"]].iloc[:-1].plot(kind="bar", figsize=(10, 6))
    plt.title("Classification Metrics (Precision, Recall, F1-Score)")
    plt.xlabel("Classes")
    plt.ylabel("Value")
    plt.grid(axis="y")
    plt.tight_layout()
    plt.savefig("classification_metrics.png")
    logging.info("Grafico delle metriche salvato come classification_metrics.png")
    plt.show()

# Caricamento e bilanciamento dei dati
file_paths = ["../Dataset misurazioni/dataset__Ticket Reserve_API.csv", "../Dataset misurazioni/dataset_Captcha.csv","../Dataset misurazioni/dataset_Train.csv","../Dataset misurazioni/datasetOrderVoucher.csv"]
data = load_and_balance_data(file_paths)

# Prepara i dati per la trasformazione
X = data[['mean', 'std_dev', 'min', 'max', 'median']].values
y = data['is_stable'].values
logging.info("Dimensioni del dataset dopo il bilanciamento: %s", X.shape)

# Suddivisione del dataset in training e test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info("Suddivisione del dataset completata: %d training, %d test", len(y_train), len(y_test))

# Generazione dei kernel casuali
num_kernels = 500
kernels = generate_random_kernels(input_length=X_train.shape[1], num_kernels=num_kernels)

# Salva i kernel generati
joblib.dump(kernels, "rocket_kernels.pkl")
logging.info("Kernel salvati come rocket_kernels.pkl")

# Trasformazione dei dati usando ROCKET
X_train_transformed = transform_with_rocket(X_train, kernels)
X_test_transformed = transform_with_rocket(X_test, kernels)
logging.info("Trasformazione dei dati completata")

# Addestramento del classificatore Ridge
classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10), scoring='accuracy', cv=5)
classifier.fit(X_train_transformed, y_train)
logging.info("Addestramento del classificatore completato")

# Salva il modello addestrato
joblib.dump(classifier, "ridge_classifier_model.pkl")
logging.info("Modello Ridge salvato come ridge_classifier_model.pkl")

# Valutazione del modello
logging.info("Valutazione del modello sui dati di test")
report, conf_matrix = evaluate_model(classifier, X_test_transformed, y_test)

# Creazione e salvataggio dei grafici
plot_metrics(report, conf_matrix)
save_metrics_as_image(report, conf_matrix)
