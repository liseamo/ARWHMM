import csv

# === Paramètres ===
fichier_csv = "/Users/lise/Downloads/mice_3530_2025-01-23-10-55-42_video_trig_0.gpio_relative.csv"
mot_cle = "GPIO-2"
seuil_presence = 40000  # fréquence IR minimale pour considérer qu'il y a détection
gap_max = 5
nb_lignes = 0
presence_actuelle = False
debut_presence = None
periodes = []

with open(fichier_csv, mode="r", encoding="utf-8", errors="ignore") as f:
    lecteur = csv.reader(f)
    for i, ligne in enumerate(lecteur, start=1):
        nb_lignes += 1

        if len(ligne) >= 3 and mot_cle in ligne[1]:
            try:
                timestamp = float(ligne[0])
                valeur = float(ligne[2])
            except ValueError:
                continue

            detection = valeur > seuil_presence

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
# Si le fichier finit en présence
if presence_actuelle:
    periodes.append((debut_presence, timestamp))

# --- Résultats ---
print(f"Nombre total de lignes : {nb_lignes}")
print(f"Nombre de périodes de présence détectées : {len(periodes_fusionnees)}")

for start, end in periodes_fusionnees[:10]:
    print(f"Présence de {start:.4f}s à {end:.4f}s (durée {(end - start):.4f}s)")









