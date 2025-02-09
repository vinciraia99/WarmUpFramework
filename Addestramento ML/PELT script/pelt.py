import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
import xml.etree.ElementTree as ET


# Funzione per determinare lo steady state con PERT
def find_steady_state(data, warm_up_percentage=0.1):
    """
    Identifica il punto di steady state utilizzando l'algoritmo PERT.

    Args:
        data (pd.Series): Serie temporale dei tempi di esecuzione.
        warm_up_percentage (float): Percentuale iniziale considerata "warm-up".

    Returns:
        int: Iterazione in cui inizia lo steady state.
    """
    warm_up_samples = int(len(data) * warm_up_percentage)
    steady_state_threshold = data[warm_up_samples:].mean()

    for i in range(warm_up_samples, len(data)):
        if data[i:].mean() <= steady_state_threshold:
            return i
    return warm_up_samples


# Funzione per leggere i dati XML di JMeter
def parse_jmeter_logs(file_path):
    """
    Legge i dati dai log XML di JMeter e li converte in un DataFrame.

    Args:
        file_path (str): Percorso del file XML.

    Returns:
        pd.DataFrame: Dati estratti con latenza (lt) e timestamp (ts).
    """
    tree = ET.parse(file_path)
    root = tree.getroot()

    data = []
    for sample in root.findall(".//httpSample"):
        ts = int(sample.get("ts"))
        lt = int(sample.get("lt"))
        rc = sample.get("rc")
        data.append({"timestamp": ts, "latency": lt, "responseCode": rc})

    return pd.DataFrame(data)


# Percorso del file XML
file_path = "../Dataset misurazioni/results_Ticket Reserve_API.xml"

# Carica i dati dai log di JMeter
data = parse_jmeter_logs(file_path)

# Filtra solo le richieste con codice di risposta 200
filtered_data = data[data['responseCode'] == "200"]

# Ordina i dati per timestamp
filtered_data = filtered_data.sort_values(by='timestamp')

# Estrai i tempi di latenza
latency_times = filtered_data['latency']  # Lascia in millisecondi

# Trova lo steady state
steady_state_start = find_steady_state(latency_times)

# Visualizza i dati
plt.figure(figsize=(10, 6))
plt.plot(filtered_data['timestamp'], latency_times, label="Latency Time", color="red")
plt.axvline(x=filtered_data['timestamp'].iloc[steady_state_start], color='black', linestyle='--',
            label="Steady State Start")
plt.title("JMeter Latency Time Analysis")
plt.xlabel("Timestamp")
plt.ylabel("Latency (ms)")
plt.legend()
plt.grid(True)
plt.show()
