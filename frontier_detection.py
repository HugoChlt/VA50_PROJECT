#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry
import threading
import signal
import sys

thread = True
map = None
costmap = None
odom = None

def signal_handler(sig, frame):
    global thread
    rospy.loginfo("Interrupt received, stopping the thread...")
    thread = False
    sys.exit()

def pixel_to_odom(pixel):
    size = 1984
    u0 = size//2
    return ((pixel[0]-u0)/20, (pixel[1]-u0)/20)

def odom_to_pixel(odom):
    size = 1984
    u0 = size//2
    return ((odom[0]*20)+u0, (odom[1]*20)+u0)

def is_pixel_near(image, pixel, target_value):
    width = image.shape[1]
    height = image.shape[0]
    x = pixel[0]
    y = pixel[1]

    if x + 1 < width:
        if image[x+1][y] == target_value:
            return True
    if x - 1 >= 0:
        if image[x-1][y] == target_value:
            return True
    if y + 1 < height:
        if image[x][y+1] == target_value:
            return True
    if y - 1 >= 0:
        if image[x][y-1] == target_value:
            return True
        
    return False

def sort_pixel_near(image, target_pixels, target_value):
    new_target_pixels = []

    if target_pixels is None:
        return None

    for pixel in target_pixels:
        if is_pixel_near(image, pixel, target_value):
            new_target_pixels.append(pixel)

    return np.array(new_target_pixels)

def find_nearest_pixel(map, start_pixel, target_value, neighbor_value):
    """
    Trouve le pixel le plus proche ayant une valeur spécifique (target_value).

    :param image: numpy.ndarray, tableau représentant l'image ou la carte
    :param start_pixel: tuple (x, y), coordonnées du pixel de départ
    :param target_value: int, valeur du pixel cible à rechercher
    :return: tuple (x, y), coordonnées du pixel trouvé ou None si aucun pixel trouvé
    """
    height, width = map.shape
    start_x, start_y = start_pixel

    # Vérification des limites du pixel de départ
    if not (0 <= start_x < height and 0 <= start_y < width):
        raise ValueError("Le pixel de départ est en dehors des limites de l'image.")

    # Initialisation de la recherche avec une distance infinie
    min_distance = float('inf')
    nearest_pixel = None

    # Parcourt tous les pixels de l'image
    for x in range(height):
        for y in range(width):
            # Vérifie si le pixel a la valeur cible
            if map[y, x] == target_value:
                # Calcule la distance cartésienne
                distance = np.sqrt((x - start_x) ** 2 + (y - start_y) ** 2)
                # Met à jour le pixel le plus proche si la distance est plus petite
                if distance < min_distance and is_pixel_near(map, (x,y), neighbor_value):
                    min_distance = distance
                    nearest_pixel = (x, y)

    return nearest_pixel

def find_pixels(image, target_value):
    # Récupérer les coordonnées de tous les pixels ayant la valeur target_value
    target_pixels = np.argwhere(image == target_value)
    
    if len(target_pixels) == 0:
        # Aucun pixel avec la valeur target_value n'est présent
        return None
    
    return target_pixels

def find_closest_pixel(pixel, target_pixels):
    """
    Trouve le pixel avec la valeur target_value le plus proche d'un pixel donné.
    
    :param image: numpy.ndarray, la carte d'occupation (OccupancyGrid convertie en tableau NumPy)
    :param target_pixel: tuple (x, y), les coordonnées du pixel de référence
    :return: tuple (x, y) des coordonnées du pixel target_value le plus proche ou None si aucun n'est trouvé
    """
    # Calculez la distance euclidienne entre le pixel cible et chaque pixel avec target_value
    distances = np.sqrt((target_pixels[:, 0] - pixel[0])**2 + 
                        (target_pixels[:, 1] - pixel[1])**2)
    
    # Trouver l'indice du pixel avec la distance minimale
    closest_index = np.argmin(distances)
    
    # Retourner les coordonnées du pixel le plus proche
    return tuple(target_pixels[closest_index])

def search_goal():
    global thread
    global map
    global costmap
    global odom

    while thread:
        if map is not None and costmap is not None and odom is not None:
            current_map = map
            current_costmap = costmap
            map_color = cv2.cvtColor(current_map, cv2.COLOR_GRAY2BGR)
            current_odom = odom
            current_pixel = odom_to_pixel(current_odom)

            # target_pixels = find_pixels(current_map, 255)
            # target_pixels = sort_pixel_near(current_map, target_pixels, 0)
            nearest_pixel = find_nearest_pixel(current_map, current_pixel, 255, 0)
            # if target_pixels is not None:
            if nearest_pixel is not None:
                # closest_pixel = find_closest_pixel(current_pixel, target_pixels)
                cv2.circle(map_color, nearest_pixel , 2, (255,0,0), 3)
                # target_pixels = find_pixels(current_costmap, 0)
                goal = find_nearest_pixel(current_costmap, nearest_pixel, 0, 0)
                if goal is not None:
                    # goal = find_closest_pixel(nearest_pixel, target_pixels)
                    goal_odom = pixel_to_odom(goal)
                    rospy.loginfo(f"Goal trouvé aux coordonnées (x: {goal_odom[0]}, y: {goal_odom[1]}) à partir de la position (x: {current_odom[0]}, y: {current_odom[1]})")
                    cv2.circle(map_color, (int(current_pixel[0]),int(current_pixel[1])) , 2, (0,255,0), 3)
                    cv2.circle(map_color, (int(goal[0]),int(goal[1])), 2, (0,0,255), 3)
                    cv2.imwrite('/tmp/goal.jpg', map_color)

    rospy.loginfo("Arrêt du thread")

class UndiscoveredZoneExplorer:
    def __init__(self):
        """Initialise le noeud ROS, abonnez-vous aux topics de cartes et configurez les publishers."""
        rospy.init_node('undiscovered_zone_explorer', anonymous=True)

        # Paramètres de l'explorateur
        self.map_topic = rospy.get_param('~map_topic', '/robot/map')
        self.costmap_topic = rospy.get_param('~costmap_topic', '/robot/move_base/global_costmap/costmap')
        self.odom_topic = rospy.get_param('~odom_topic', "/robot/dlo/odom_node/odom")

        # Abonnements aux topics
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback)
        rospy.Subscriber(self.costmap_topic, OccupancyGrid, self.costmap_callback)
        rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback)

        # Stockage des dernières cartes reçues
        self.latest_map = None
        self.latest_costmap = None
        self.latest_odom = None

    def map_callback(self, msg):
        """Callback appelé à chaque réception de la carte principale."""
        global map

        rospy.loginfo("Nouvelle carte principale reçue, analyse des zones non découvertes...")

        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        data = np.array(msg.data).reshape((height, width))

        # Créez une image OpenCV avec les différentes catégories
        image = np.array(data, dtype=np.uint8)
        self.latest_map = image
        map = image
        
        cv2.imwrite('/tmp/map_data.jpg', image)

    def costmap_callback(self, msg):
        """Callback appelé à chaque réception de la costmap globale."""
        global costmap

        rospy.loginfo("Nouvelle costmap globale reçue, analyse des zones non découvertes...")

        width = msg.info.width
        height = msg.info.height
        resolution = msg.info.resolution
        origin_x = msg.info.origin.position.x
        origin_y = msg.info.origin.position.y

        data = np.array(msg.data).reshape((height, width))

        # Créez une image OpenCV avec les différentes catégories
        image = np.array(data, dtype=np.uint8)
        self.latest_costmap = image
        costmap = image

        cv2.imwrite('/tmp/costmap_data.jpg', image)

    def odom_callback(self, msg):
        """Callback appelé à chaque réception des données d'odométrie."""
        global odom

        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation
        # rospy.loginfo(f"Position actuelle : x={position.x}, y={position.y}, z={position.z}")
        # rospy.loginfo(f"Orientation actuelle : x={orientation.x}, y={orientation.y}, z={orientation.z}, w={orientation.w}")

        self.latest_odom = (position.x, position.y)
        odom = (position.x, position.y)

    def process_map(self, image, file_prefix, description):
        """Traite et enregistre une carte reçue (OccupancyGrid)."""
        # Enregistrer les données de la carte dans un fichier texte
        cv2.imwrite(f'{file_prefix}.jpg', image)

    def run(self):
        """Lance la boucle principale ROS"""
        rospy.loginfo("Explorateur de zones non découvertes actif.")
        rospy.spin()

if __name__ == '__main__':
    try:
        signal.signal(signal.SIGINT, signal_handler)
        thread = threading.Thread(target=search_goal)
        thread.start()
        explorer = UndiscoveredZoneExplorer()
        explorer.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Arrêt de l'explorateur de zones non découvertes.")
        thread = False
