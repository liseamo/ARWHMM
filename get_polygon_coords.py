import cv2

# === MODIFIE ICI AVEC TON CHEMIN VERS LA VIDÉO ===
video_path = r"C:\Users\lamodeo\Desktop\videos\3544-21-01-25.mp4"

# === CHARGER LA VIDÉO ===
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
cap.release()

if not ret:
    print("❌ Impossible de lire la vidéo.")
    exit()

# === LISTE POUR STOCKER LES POINTS CLIQUÉS ===
points = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"🟢 Point ajouté : ({x}, {y})")
        # Dessine un petit cercle sur le point
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.imshow("Clique pour définir le polygone", frame)

# === AFFICHER LA FRAME ET ATTENDRE LES CLICS ===
cv2.imshow("Clique pour définir le polygone", frame)
cv2.setMouseCallback("Clique pour définir le polygone", click_event)
print("📌 Clique sur les coins de ta zone, puis appuie sur une touche pour terminer.")
cv2.waitKey(0)
cv2.destroyAllWindows()

print("\n✅ Coordonnées du polygone :")
print("zone_polygon = Polygon([")
for pt in points:
    print(f"    {pt},")
print("])")
