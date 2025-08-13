import csv

# === Paramètres ===
fichier_csv = "/Users/lise/Desktop/IR_binaire/pasbisnaire/m3530/session7/mice_3530_2025-01-27-11-33-03_video_trig_0.gpio_relative.csv"
fichier_sortie = "/Users/lise/Desktop/IR_binaire/3530/IR_3530_session7_binaire.csv"
mot_cle = "GPIO-2"
seuil_presence = 30000  # fréquence IR minimale pour considérer qu'il y a détection
gap_max = 5
nb_lignes = 0
presence_actuelle = False
debut_presence = None
periodes = []
timestamps = []
valeurs = []
binaire = []

with open(fichier_csv, mode="r", encoding="utf-8", errors="ignore") as f:
    lecteur = csv.reader(f)
    for i, ligne in enumerate(lecteur, start=1):
        nb_lignes += 1

        if len(ligne) >= 3 and mot_cle in ligne[1]:
            try:
                timestamp = float(ligne[0])
                valeur = float(ligne[2])
                timestamps.append(timestamp)
                valeurs.append(valeur)  
            except ValueError:
                continue

            detection = valeur > seuil_presence
            binaire.append(1 if detection else 0)

            # Passage de absence → présence
            if detection and not presence_actuelle:
                presence_actuelle = True
                debut_presence = timestamp

            # Passage de présence → absence
            elif not detection and presence_actuelle:
                presence_actuelle = False
                periodes.append((debut_presence, timestamp))

# --- Étape 4 : fusionner les périodes proches ---
periodes_fusionnees = []
for start, end in periodes:
    if not periodes_fusionnees:
        periodes_fusionnees.append([start, end])
    else:
        dernier_start, dernier_end = periodes_fusionnees[-1]
        if start - dernier_end <= gap_max:
            # fusionner
            periodes_fusionnees[-1][1] = end
        else:
            periodes_fusionnees.append([start, end])
# # Filtrer pour ne garder que les épisodes de plus de 2 secondes
# periodes_fusionnees = [
#     (start, end) for start, end in periodes_fusionnees
#     if (end - start) > 2
# ]

# Si le fichier finit en présence
if presence_actuelle:
    periodes.append((debut_presence, timestamp))
with open(fichier_sortie, mode="w", newline="") as fout:
    writer = csv.writer(fout)
    writer.writerow(["timestamp", "valeur_IR", "presence_binaire"])
    for t, v, b in zip(timestamps, valeurs, binaire):
        writer.writerow([t, v, b])

# --- Résultats ---
print(f"Nombre total de lignes : {nb_lignes}")
print(f"Nombre de périodes de présence détectées : {len(periodes_fusionnees)}")

for start, end in periodes_fusionnees[:10]:
    print(f"Présence de {start:.4f}s à {end:.4f}s (durée {(end - start):.4f}s)")








