# WarmUpFramework

## Descrizione del Progetto

WarmUpFramework è un progetto sviluppato per il preprocessing e l'addestramento di modelli di machine learning con tecniche avanzate. Il framework include strumenti per la conversione di dataset, la misurazione delle performance, l'addestramento di reti neurali e l'esposizione di modelli tramite API.

Il progetto è stato sviluppato con l'obiettivo di identificare automaticamente lo **steady state** nei sistemi a microservizi. Per farlo, utilizza tecniche avanzate di **segmentazione di serie temporali**, come l'algoritmo **PELT (Pruned Exact Linear Time)**, e modelli di **machine learning**, tra cui **FCN (Fully Convolutional Network)** e **ROCKET (Random Convolutional Kernel Transform)**. 

Questa ricerca è stata sviluppata come parte della tesi di laurea, che esplora metodi per automatizzare il performance testing attraverso l'analisi delle metriche di latenza e throughput, distinguendo tra la fase di **warm-up** e quella di **steady state**. Il progetto è stato validato utilizzando il benchmark **Train Ticket**, un sistema composto da **41 microservizi**, che simula un'applicazione reale di prenotazione di biglietti ferroviari.

## Requisiti

Per eseguire il progetto, assicurati di avere installato:

- Python 3.8+
- Docker & Docker Compose
- Librerie Python necessarie (elencate in `requirements.txt`)

## Installazione

Clona il repository e installa le dipendenze richieste:

```bash
git clone https://github.com/tuo-repo/WarmUpFramework.git
cd WarmUpFramework
pip install -r requirements.txt
```

## Dataset

### Dataset Misurazioni
Il dataset contenuto nella cartella `Dataset misurazioni` raccoglie i risultati delle misurazioni dei modelli addestrati. Questi file XML contengono metriche dettagliate sulle prestazioni e sull'accuratezza.

### Conversione in blocchi PELT
Il modulo `PELT script` contiene lo script `pelt.py`, che viene utilizzato per suddividere i dataset in blocchi tramite l'algoritmo PELT, migliorando così l'efficacia del preprocessing dei dati. Questo processo consente di individuare il punto di transizione tra warm-up e steady state, ottimizzando i dati di input per l'addestramento del modello.

## Addestramento dei Modelli

Il framework include diversi script per l'addestramento di modelli di machine learning, con implementazioni di **reti neurali convoluzionali (FCN)** e modelli basati su **kernel casuali (ROCKET)**.

Per avviare l'addestramento, eseguire i seguenti comandi:

```bash
python Addestramento\ ML/Script\ di\ addestramento\ modelli\ ML/FCN_multi.py
python Addestramento\ ML/Script\ di\ addestramento\ modelli\ ML/rocket.py
```

L'addestramento è basato su dataset preprocessati, che vengono segmentati attraverso il metodo PELT per ottenere una maggiore precisione nei risultati.

## Uso delle API via Container

Il framework include un'API che consente di interrogare i modelli addestrati. L'API è containerizzata per facilitare il deployment e può essere avviata facilmente con Docker.

Per avviare l'API tramite Docker, eseguire:

```bash
docker-compose up --build
```

In alternativa, è possibile avviare manualmente l'API eseguendo:

```bash
python Container\ API\ ML/api.py
```

L'API permette di inviare dati ai modelli e ottenere previsioni, rendendo il sistema adatto all'integrazione in applicazioni reali che richiedono analisi predittive in tempo reale.

### Train Ticket e Docker
Il progetto è stato testato all'interno di un ambiente **Docker**, garantendo esecuzioni riproducibili e isolando il framework dal sistema host. Per la validazione sperimentale è stato utilizzato il benchmark **Train Ticket**, che simula una piattaforma reale di prenotazione di biglietti ferroviari con **41 microservizi**. Questo sistema è disponibile su GitHub: [Train Ticket Docker](https://github.com/vinciraia99/train-ticket-docker).

## Contributi

Se vuoi contribuire al progetto, crea una nuova branch e invia una pull request con le modifiche. Sono benvenuti miglioramenti alle pipeline di preprocessing, nuovi modelli di machine learning e ottimizzazioni delle API.

## Autore

Questo progetto è stato sviluppato come parte della tesi di laurea di Vincenzo Raia.

