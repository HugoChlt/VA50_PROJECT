#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

import time  # Import pour utiliser la fonction sleep

class UndiscoveredZoneExplorer:
    def __init__(self):
        rospy.init_node('undiscovered_zone_explorer', anonymous=True)
        
        # Paramètres de l'explorateur
        self.map_topic = rospy.get_param('~map_topic', '/robot/map')
        self.costmap_topic = rospy.get_param('~costmap_topic', '/robot/move_base/global_costmap/costmap')
        self.goal_topic = rospy.get_param('~goal_topic', '/robot/move_base_simple/goal')
        
        # Initialisation du publisher pour envoyer les objectifs à move_base
        self.goal_publisher = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=10)
        
        # Variables pour stocker les messages reçus
        self.latest_map = None
        self.latest_costmap = None
        self.subscribers_active = True  # Drapeau pour gérer les abonnements
        self.is_moving = False

        # Abonnements aux topics
        self.map_sub = rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback)
        self.costmap_sub = rospy.Subscriber(self.costmap_topic, OccupancyGrid, self.costmap_callback)

    def map_callback(self, map_msg):
        """Callback pour le topic de la carte /map."""
        if not self.subscribers_active:
            return
        rospy.loginfo("Message reçu sur /map.")
        self.latest_map = map_msg
        self.check_and_process()

    def costmap_callback(self, costmap_msg):
        """Callback pour le topic de la carte /costmap."""
        if not self.subscribers_active:
            return
        rospy.loginfo("Message reçu sur /costmap.")
        self.latest_costmap = costmap_msg
        self.check_and_process()

    def check_and_process(self):
        """Vérifie si les deux messages ont été reçus et traite les cartes."""
        if self.latest_map is not None and self.latest_costmap is not None:
            rospy.loginfo("Les deux messages sont disponibles, traitement en cours.")
            if not self.is_moving:
                self.subscribers_active = False  # Désactiver les abonnements
                self.map_sub.unregister()  # Arrêter d'écouter le topic /map
                self.costmap_sub.unregister()  # Arrêter d'écouter le topic /costmap
                self.find_and_send_goal()

    def reactivate_subscribers_after_delay(self):
        """Réactive les abonnements après un délai."""
        rospy.loginfo("Pause de 30 secondes pour permettre au robot d'atteindre sa destination.")
        time.sleep(30)  # Pause de 30 secondes
        rospy.loginfo("Réactivation des abonnements pour continuer l'exploration.")
        self.subscribers_active = True
        self.map_sub = rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback)
        self.costmap_sub = rospy.Subscriber(self.costmap_topic, OccupancyGrid, self.costmap_callback)
        self.is_moving = False

    def find_and_send_goal(self):
        """Trouve la prochaine zone inconnue à explorer et envoie un objectif à move_base."""
        if not self.latest_map or not self.latest_costmap:
            rospy.logwarn("Les cartes nécessaires ne sont pas encore disponibles.")
            return

        rospy.loginfo("Traitement")
        # Traitement de la carte /map
        map_data, map_info = self.process_occupancy_grid(self.latest_map)
        unknown_indices = self.find_unknown_zones(map_data)

        if len(unknown_indices) == 0:
            rospy.loginfo("Aucune zone inconnue détectée dans la carte.")
            return

        # Rechercher les pixels connus proches des zones inconnues
        for i, j in unknown_indices:
            nearest_known = self.find_adjacent_known_pixel(i, j, map_data)
            if nearest_known:
                x, y = self.map_to_costmap_coordinates(nearest_known, map_info, self.latest_costmap.info)
                final_target = self.find_nearest_free_pixel_in_costmap(x, y, self.latest_costmap)
                if final_target:
                    self.send_goal(final_target[0], final_target[1])
                    return

        rospy.logwarn("Aucune zone accessible n'a pu être identifiée.")

    def process_occupancy_grid(self, grid):
        """Convertit un message OccupancyGrid en numpy array et extrait les informations de la carte"""
        width = grid.info.width
        height = grid.info.height
        resolution = grid.info.resolution
        origin = grid.info.origin.position

        data = np.array(grid.data).reshape((height, width))
        return data, {"resolution": resolution, "origin": (origin.x, origin.y)}

    def find_unknown_zones(self, data):
        """Trouve les indices des pixels inconnus (-1) adjacents à des zones connues (0)"""
        unknown_indices = []
        height, width = data.shape

        for i in range(height):
            for j in range(width):
                if data[i, j] == -1:  # Pixel inconnu
                    # Check boundaries before accessing neighbors
                    if (
                        (i > 0 and j > 0 and data[i-1, j-1] == 0) or
                        (i > 0 and data[i-1, j] == 0) or
                        (i > 0 and j < width - 1 and data[i-1, j+1] == 0) or
                        (j > 0 and data[i, j-1] == 0) or
                        (j < width - 1 and data[i, j+1] == 0) or
                        (i < height - 1 and j > 0 and data[i+1, j-1] == 0) or
                        (i < height - 1 and data[i+1, j] == 0) or
                        (i < height - 1 and j < width - 1 and data[i+1, j+1] == 0)
                    ):
                        unknown_indices.append((i, j))

        return unknown_indices

    def get_adjacent_indices(self, i, j, height, width):
        """Renvoie les indices des pixels adjacents"""
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        return [(i + di, j + dj) for di, dj in directions if 0 <= i + di < height and 0 <= j + dj < width]

    def find_adjacent_known_pixel(self, i, j, data):
        """Trouve un pixel connu (0) adjacent au pixel donné (-1)"""
        for ni, nj in self.get_adjacent_indices(i, j, *data.shape):
            if data[ni, nj] == 0:
                return (ni, nj)
        return None

    def map_to_costmap_coordinates(self, pixel, map_info, costmap_info):
        """Transpose les coordonnées d'un pixel de la carte vers le costmap"""
        map_res, (map_origin_x, map_origin_y) = map_info["resolution"], map_info["origin"]
        costmap_res, (costmap_origin_x, costmap_origin_y) = costmap_info.resolution, (costmap_info.origin.position.x, costmap_info.origin.position.y)

        x = map_origin_x + pixel[1] * map_res
        y = map_origin_y + pixel[0] * map_res

        j = int((x - costmap_origin_x) / costmap_res)
        i = int((y - costmap_origin_y) / costmap_res)

        return i, j

    def find_nearest_free_pixel_in_costmap(self, i, j, costmap):
        """Trouve le pixel libre (valeur 0) le plus proche dans le costmap"""
        costmap_data = np.array(costmap.data).reshape((costmap.info.height, costmap.info.width))
        if costmap_data[i, j] == 0:
            return self.pixel_to_world_coordinates(i, j, costmap.info)

        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < costmap_data.shape[0] and 0 <= nj < costmap_data.shape[1]:
                if costmap_data[ni, nj] == 0:
                    return self.pixel_to_world_coordinates(ni, nj, costmap.info)

        return None

    def pixel_to_world_coordinates(self, i, j, info):
        """Convertit des indices de pixel en coordonnées mondiales"""
        x = info.origin.position.x + j * info.resolution
        y = info.origin.position.y + i * info.resolution
        return x, y

    def send_goal(self, x, y):
        """Crée et publie un objectif PoseStamped à move_base."""
        self.is_moving = True
        goal = PoseStamped()
        goal.header.frame_id = "robot_map"
        goal.header.stamp = rospy.Time.now()

        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1

        rospy.loginfo(f"Envoi d'un objectif à x={x}, y={y}")
        self.goal_publisher.publish(goal)

        # Attendre un délai avant de réactiver les abonnements
        self.reactivate_subscribers_after_delay()

    def run(self):
        """Lance la boucle principale ROS"""
        rospy.loginfo("Explorateur de zones non découvertes actif.")
        rospy.spin()

if __name__ == '__main__':
    try:
        explorer = UndiscoveredZoneExplorer()
        explorer.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("Arrêt de l'explorateur de zones non découvertes.")