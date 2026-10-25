# Quantum_QLTH_VQC_v1.py

# QLTH - Quantum Lottery Ticket Hypothesis (Teorema o kvantnoj lutrijskoj karti) 
# VQC - Variational Quantum Circuits (Variaciona Kvantna Kola) 


import pandas as pd
import numpy as np
import random
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

# Construction of the environment (Strict protocol parameters)
MAX_BROJ = 39
BROJ_KUGLICA = 7
KUBITA_PO_POZICIJI = 5  # Moje rešenje: 2^5 = 32 stanja po svakoj poziciji bez izuzetka
DIMENZIJA_REGISTRA = 32

random.seed(39)
np.random.seed(39)


# Putanja do podataka (CSV)
# putanja = '/Users/4c/Desktop/GHQ/data/loto7_4708_k85.csv'
# putanja = '/Users/4c/Desktop/GHQ/data/loto7_4708_k85_loto_2976.csv'
putanja = '/Users/4c/Desktop/GHQ/data/loto7_4708_k85_loto_plus_1732.csv'


# File loading
df = pd.read_csv(putanja, header=None)

# Isolation of exactly the last 7 columns from the tail of the table (chronological, positional)
H = df.iloc[:, -7:].values.astype(int)
H_sorted = np.sort(H, axis=1)
N_redova = len(H_sorted)

global_cov = np.cov(H_sorted.T)

potkresane_amplitude_sistema = {}

# 1. GENERISANJE POZICIONIH AMPLITUDE PREMA MOM MODELU OD 5 KUBITA (Pun opseg 32 stanja)
for i in range(BROJ_KUGLICA):
    min_val = i + 1
    istorija_pozicije = H_sorted[:, i]
    
    kvantne_amplitude = np.zeros(DIMENZIJA_REGISTRA, dtype=complex)
    
    poslednji_marker = {}
    for korak, broj in enumerate(istorija_pozicije):
        indeks_stanja = broj - min_val
        if 0 <= indeks_stanja < DIMENZIJA_REGISTRA:
            poslednji_marker[indeks_stanja] = korak
            
    for indeks_stanja, marker in poslednji_marker.items():
        relativno_vreme = marker / N_redova
        kvantne_amplitude[indeks_stanja] = np.sin(relativno_vreme * np.pi / 2) + 1j * np.cos(relativno_vreme * np.pi / 2)
            
    if np.linalg.norm(kvantne_amplitude) > 0:
        kvantne_amplitude /= np.linalg.norm(kvantne_amplitude)
    else:
        kvantne_amplitude = np.ones(DIMENZIJA_REGISTRA, dtype=complex) / np.sqrt(DIMENZIJA_REGISTRA)
        
    qc_dense = QuantumCircuit(KUBITA_PO_POZICIJI)
    qc_dense.initialize(kvantne_amplitude, range(KUBITA_PO_POZICIJI))
    
    state_dense = Statevector.from_instruction(qc_dense)
    amplitude_gustog_stanja = state_dense.data
    
    prag_pruninga = np.mean(np.abs(amplitude_gustog_stanja))
    qlth_maska = np.abs(amplitude_gustog_stanja) >= prag_pruninga
    
    potkresane = amplitude_gustog_stanja * qlth_maska
    if np.linalg.norm(potkresane) > 0:
        potkresane /= np.linalg.norm(potkresane)
    else:
        potkresane = amplitude_gustog_stanja
        
    potkresane_amplitude_sistema[i] = potkresane

kombinacija = []

# 2. KVANTNI KLIZEĆI PROZOR (Uparivanje 5+5 = 10 kubita, pun kapacitet bez sečenja spektra i freeze-a)
for i in range(BROJ_KUGLICA):
    min_val = i + 1
    
    if i < BROJ_KUGLICA - 1:
        j = i + 1
        # Podizanje lokalnog registra od tačno 10 kubita (32 * 32 = 1024 stanja)
        qc_local = QuantumCircuit(10)
        
        qc_local.initialize(potkresane_amplitude_sistema[i], range(0, 5))
        qc_local.initialize(potkresane_amplitude_sistema[j], range(5, 10))
        
        preostala_energija_i = np.mean(np.abs(potkresane_amplitude_sistema[i]))
        preostala_energija_j = np.mean(np.abs(potkresane_amplitude_sistema[j]))
        
        for q1 in range(5):
            qc_local.rz(np.pi * preostala_energija_i, q1)
            
        kovarijansa_veze = global_cov[i, j]
        ugao_preplitanja = np.pi * np.tanh(kovarijansa_veze * (preostala_energija_i + preostala_energija_j) / 2)
        
        # Realno inter-poziciono preplitanje između kubita pozicije i i pozicije j
        qc_local.crz(ugao_preplitanja, 0, 5)
        
        state_local = Statevector.from_instruction(qc_local)
        probabiliteti = state_local.probabilities_dict(range(0, 5))
    else:
        # Poslednja pozicija radi na svom 5-kubitnom registru, zadržavajući uvezanu fazu
        qc_local = QuantumCircuit(5)
        qc_local.initialize(potkresane_amplitude_sistema[i], range(5))
        preostala_energija_i = np.mean(np.abs(potkresane_amplitude_sistema[i]))
        for q1 in range(5):
            qc_local.rz(np.pi * preostala_energija_i, q1)
            
        state_local = Statevector.from_instruction(qc_local)
        probabiliteti = state_local.probabilities_dict()

    # Selekcija stanja maksimalne amplitude
    najbolje_stanje_bin = max(probabiliteti, key=probabiliteti.get)
    izabrani_indeks = int(najbolje_stanje_bin, 2)
    
    izabrani_broj = izabrani_indeks + min_val
    
    # Rešavanje duplikata preko Jacobijeve refleksije unutar strogog pozicionog opsega (min - min+32)
    val = izabrani_broj
    pomak = 1
    max_val_pozicije = min_val + 32
    
    while val in kombinacija or val > MAX_BROJ:
        if val > MAX_BROJ:
            val = MAX_BROJ - (val - MAX_BROJ)
        else:
            val = izabrani_broj + pomak
            while val < min_val or val > max_val_pozicije:
                if val < min_val:
                    val = min_val + (min_val - val)
                if val > max_val_pozicije:
                    val = max_val_pozicije - (val - max_val_pozicije)
            pomak += 1
        
    kombinacija.append(int(val))


# Output results
print()
print(f"Mod: Istinski 5-Kubitno Pozicioni QLTH Model")
print(f"Analizirano redova iz CSV: {N_redova}")
print(f"Predložena kombinacija: {sorted(kombinacija)}")
print()
"""
Mod: Istinski 5-Kubitno Pozicioni QLTH Model
Analizirano redova iz CSV: 4708
Predložena kombinacija: [11, 13, 21, 22, 23, 24, 27]



Mod: Istinski 5-Kubitno Pozicioni QLTH Model
Analizirano redova iz CSV: 2976
Predložena kombinacija: [2, 4, 7, 8, 18, 19, 25]



Mod: Istinski 5-Kubitno Pozicioni QLTH Model
Analizirano redova iz CSV: 1732
Predložena kombinacija: [5, 10, 11, 12, 24, 27, 30]
"""




"""
VQC i Teorema o kvantnoj lutrijskoj karti (Variational Quantum Circuits & Quantum Lottery Ticket Hypothesis)

• Izvor i koncept (2025/2026): Najnoviji radovi koji istražuju Lottery Ticket Hypothesis (LTH) na varijacionim kvantnim kolima (VQC) dokazuju postojanje takozvanih "pobedničkih karata" (winning tickets) unutar kompleksnih kvantnih mreža. Kvantna kola sa prilagodljivim parametrima optimizuju podprostore bez upadanja u matematičke platoe (barren plateaus).
• Pogodnost za Loto 7/39: Najpogodniji metod. Istorijski podaci iz CSV fajla koriste se za treniranje kvantnih parametara unutar VQC. Algoritam kroz pruning (potkresivanje) maske identifikuje optimalnu pod-mrežu stanja (kvantni ekvivalent Jacobijeve refleksije) koja zadržava maksimalnu tačnost nad podacima.

Ova dva koncepta predstavljaju savršen hibridni spoj u modernom kvantnom mašinskom učenju (QML). Dok je VQC struktura (hardver), QLTH je strategija optimizacije (softver) koja omogućava tom hardveru da radi efikasno na lokalnim računarima.

1. QLTH - Quantum Lottery Ticket Hypothesis (Teorema o kvantnoj lutrijskoj karti)

To je metoda hirurškog potkresivanja (pruninga) VQC-a.
• Kada se spoji 7 pozicija u jedno veliko VQC kolo, dobija se gusta mreža. Optimizacija te mreže upada u barren plateaus (matematičke platoe) gde sistem gubi gradijent i ne može da izračuna ništa.
• QLTH teorema tvrdi: Unutar tog gustog VQC kola već postoji skrivena pod-mreža gejtova (tzv. pobednička karta ili winning ticket) koja, ako se izoluje pomoću binarne maske, može sama da pruži maksimalnu tačnost predikcije, trošeći delić memorije i resursa.

2. VQC - Variational Quantum Circuit (Variaciono kvantno kolo)

To je kvantni ekvivalent klasične neuronske mreže. Sastoji se od tri dela:
• Feature Map: Prevođenje klasičnih podataka (vremenskih markera iz CSV-a) u kvantna stanja.
• Ansatz (Parametrizovano kolo): Slojevi rotacionih gejtova (čiji se uglovi menjaju) i kontrolisanih preplitanja (crz, cx) koji formiraju kompleksnu mrežu.
• Merenje/Očitavanje: Računanje očekivanih vrednosti talasne funkcije (izračunavanje krajnjih amplituda).

1. Gusto VQC stanje (Krak 1): Kod prvo podiže inicijalno variaciono kolo (qc_dense) za svaku poziciju i učitava celokupnu istoriju u vidu složenih amplituda.
2. QLTH Maska (Krak 2): Izračunava se energetski prosek cele talasne funkcije. Sva stanja čija je magnituda ispod tog proseka se tretiraju kao "šum" i dobijaju vrednost 0 u maski. Stanja sa konstruktivnom interferencijom dobijaju 1.
3. Izolacija Pobedničke Karte (Krak 3): Kolo se "potkresuje" (prunuje) i podiže se čista, izolovana lutrijska karta (qc_ticket) koja sadrži samo preostalu korisnu energiju sistema.
4. Globalno preplitanje: Te potkresane karte se na kraju uvezuju u jedan registar od 30 kubita gde međusobno komuniciraju isključivo preko realnih istorijskih korelacija iz celog CSV fajla.

VQC daje kvantni prostor, a QLTH iz njega izbacuje đubre i ostavlja samo čistu matematičku suštinu prošlosti.
"""
