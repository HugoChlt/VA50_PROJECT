import cv2 as cv
import numpy as np


def file_opening():
    # Chemin de l'image
    image_path = "fullmap.png"
    I = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

    if I is not None:
        if image_path.lower().endswith('.png'):
            print("Image PNG chargée avec succès.")
            cv.imshow("2D map", I)
            cv.waitKey(0)
            cv.destroyAllWindows()
        else:
            print("Erreur : Le fichier chargé n'a pas l'extension '.png'.")
    else:
        print("Erreur : L'image n'a pas pu être chargée.")

    return I


def map_filtering(I):
    # Appliquer un flou gaussien
    I = cv.GaussianBlur(I, (5, 5), 0)

    # Appliquer une binarisation avec la méthode d'Otsu
    ret, th = cv.threshold(I, 254, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)

    # Créer un noyau pour les opérations morphologiques
    kernel = np.ones((3, 3), np.uint8)
    I_filtered = cv.morphologyEx(th, cv.MORPH_OPEN, kernel, iterations=2)

    # Enregistrer la carte filtrée
    cv.imwrite('map_filtered.png', I_filtered)

    # Afficher la carte filtrée
    cv.imshow("2D map", I_filtered)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return I_filtered


def exit_detection(I):
    # Détection des contours
    contours, hierarchy = cv.findContours(I, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Convertir l'image en couleur pour dessiner les résultats
    I_color = cv.cvtColor(I, cv.COLOR_GRAY2BGR)

    # Liste pour stocker les positions des fins de murs
    wall_endings = []

    # Analyse des contours
    for cnt in contours:
        if len(cnt) > 5:  # Filtrer les petits contours
            # Approximation du contour pour réduire les points inutiles
            epsilon = 0.01 * cv.arcLength(cnt, True)
            approx = cv.approxPolyDP(cnt, epsilon, True)

            # Analyse des points approximés
            for i in range(1, len(approx)):
                # Calculer la distance entre les points consécutifs
                dist = np.linalg.norm(approx[i][0] - approx[i - 1][0])
                if dist > 9:  # Seuil pour détecter une discontinuité
                    wall_endings.append(tuple(approx[i][0]))

    # Filtrer les wall_endings espacés de moins de 8 pixels
    filtered_endings = []
    for i, point1 in enumerate(wall_endings):
        too_close = False
        for j, point2 in enumerate(wall_endings):
            if i != j:
                dist = np.linalg.norm(np.array(point1) - np.array(point2))
                if dist < 8:  # Seuil de 8 pixels
                    too_close = True
                    break
        if not too_close:
            filtered_endings.append(point1)

    # Dessiner les contours
    cv.drawContours(I_color, contours, -1, (0, 255, 0), 1)

    # Relier les points de filtered_endings les plus proches entre eux
    for i, point1 in enumerate(filtered_endings):
        min_dist = float('inf')
        closest_point = None
        for j, point2 in enumerate(filtered_endings):
            if i != j:
                dist = np.linalg.norm(np.array(point1) - np.array(point2))
                if dist < min_dist:
                    min_dist = dist
                    closest_point = point2

        if closest_point is not None:
            # Dessiner une ligne entre point1 et closest_point
            cv.line(I_color, point1, closest_point, (255, 0, 0), 2)  # Lignes en bleu
            # Calculer le point médian entre point1 et closest_point
            mid_point = ((point1[0] + closest_point[0]) // 2, (point1[1] + closest_point[1]) // 2)
            cv.circle(I_color, mid_point, 5, (0, 255, 0), -1)  # Marquer le point médian en vert
            # Afficher les coordonnées du point médian
            cv.putText(I_color, f"({mid_point[0]}, {mid_point[1]})", mid_point, cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 0), 1)

    cv.imwrite('map_detections.png', I_color)
    # Afficher les résultats
    cv.imshow("Wall Endings", I_color)
    cv.waitKey(0)
    cv.destroyAllWindows()

    return filtered_endings


def main():
    map = file_opening()
    if map is not None:
        map_filtered = map_filtering(map)
        exit_detection(map_filtered)


if __name__ == "__main__":
    main()
