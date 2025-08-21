# session4:GPIO2 + GPIO3
# session6:GPIO2 + GPIO3
# session7:GPIO2 + GPIO3
#mice3157
# session 3: GPIO2 + GPIO 4
# session 5: GPIO2 + GPIO 4
# session 6: GPIO2 + GPIO 4
#pour mice  m3530, m3544, m3546
#mice3157session2 : GPIO 2 





import csv

# === Paramètres ===
fichier_csv = r"C:\Users\lamodeo\Desktop\gpio_binaire\pas_binaire\m3157\session6\3157_2024-05-27-15-05-40_video_trig_0.gpio_relative.csv"
fichier_sortie = r"C:\Users\lamodeo\Desktop\gpio_binaire\3157\IR_3157_session6_binaire_aieaieaie.csv"

mot_cle_2 = "GPIO-2"
mot_cle_3 = "GPIO-4"
seuil_presence = 30000  # seuil IR
gap_max = 4             # fusion des périodes proches
nb_lignes = 0
presence_actuelle = False
debut_presence = None
periodes = []
timestamps = []
valeurs = []
binaire = []

gpio2_data = {}
gpio3_data = {}

# === Lecture du CSV ===
with open(fichier_csv, mode="r", encoding="utf-8", errors="ignore") as f:
    lecteur = csv.reader(f)
    for i, ligne in enumerate(lecteur, start=1):
        nb_lignes += 1
        if len(ligne) >= 3:
            try:
                timestamp = float(ligne[0])
                valeur = float(ligne[2])
            except ValueError:
                continue

            if mot_cle_2 in ligne[1]:
                gpio2_data[timestamp] = valeur
            elif mot_cle_3 in ligne[1]:
                gpio3_data[timestamp] = valeur

# === Fusion des timestamps et détection combinée ===
all_timestamps = sorted(set(gpio2_data.keys()) | set(gpio3_data.keys()))
for t in all_timestamps:
    val2 = gpio2_data.get(t, 0)
    val3 = gpio3_data.get(t, 0)
    detection = (val2 > seuil_presence) or (val3 > seuil_presence)

    timestamps.append(t)
    valeurs.append(max(val2, val3))  # valeur max des deux GPIO
    binaire.append(1 if detection else 0)

    # Passage absence → présence
    if detection and not presence_actuelle:
        presence_actuelle = True
        debut_presence = t
    elif not detection and presence_actuelle:
        presence_actuelle = False
        periodes.append((debut_presence, t))

# === Fusion des périodes proches ===
periodes_fusionnees = []
for start, end in periodes:
    if not periodes_fusionnees:
        periodes_fusionnees.append([start, end])
    else:
        dernier_start, dernier_end = periodes_fusionnees[-1]
        if start - dernier_end <= gap_max:
            periodes_fusionnees[-1][1] = end
        else:
            periodes_fusionnees.append([start, end])

# === Filtrer les épisodes > 2 secondes ===
periodes_fusionnees = [
    (start, end) for start, end in periodes_fusionnees if (end - start) > 2
]

# Si le fichier finit en présence
if presence_actuelle:
    periodes_fusionnees.append((debut_presence, all_timestamps[-1]))

# === Écriture CSV ===
with open(fichier_sortie, mode="w", newline="") as fout:
    writer = csv.writer(fout)
    writer.writerow(["timestamp", "valeur_IR", "presence_binaire"])
    for t, v, b in zip(timestamps, valeurs, binaire):
        writer.writerow([t, v, b])

# === Résultats ===
print(f"Nombre total de lignes : {nb_lignes}")
print(f"Nombre de périodes de présence détectées : {len(periodes_fusionnees)}")
for start, end in periodes_fusionnees[:10]:
    print(f"Présence de {start:.4f}s à {end:.4f}s (durée {(end - start):.4f}s)")



