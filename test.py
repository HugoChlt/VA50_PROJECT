#!/usr/bin/env python3

import rospy
import numpy as np
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

class UndiscoveredZoneExplorer:
    def __init__(self):
        """Initialise le noeud ROS, abonnez-vous au topic de la carte et configurez le publisher pour move_base."""
        rospy.init_node('undiscovered_zone_explorer', anonymous=True)
        
        # Paramètres de l'explorateur
        self.costmap_topic = rospy.get_param('~costmap_topic', '/robot/move_base/global_costmap/costmap')
        self.goal_topic = rospy.get_param('~goal_topic', '/robot/move_base_simple/goal')
        
        # Initialisation du publisher pour envoyer les objectifs à move_base
        self.goal_publisher = rospy.Publisher(self.goal_topic, PoseStamped, queue_size=10)
        
        # Abonnement au topic costmap
        rospy.Subscriber(self.costmap_topic, OccupancyGrid, self.costmap_callback)
        
        # Stockage de la dernière carte reçue
        self.latest_costmap = None
        
        # Indique si le robot est en mouvement ou non
        self.is_moving = False

    def costmap_callback(self, msg):
        """Callback appelé à chaque réception de la carte du costmap"""
        self.latest_costmap = msg
        rospy.loginfo("Nouvelle costmap reçue, analyse des zones non découvertes...")
        if not self.is_moving:
            self.find_and_send_goal()

    def find_and_send_goal(self):
        """Trouve la prochaine zone non découverte et l'envoie à move_base comme objectif"""
        if not self.latest_costmap:
            rospy.logwarn("Aucune carte disponible pour l'instant.")
            return

        # Récupérer les dimensions de la carte
        width = self.latest_costmap.info.width
        height = self.latest_costmap.info.height
        resolution = self.latest_costmap.info.resolution
        origin_x = self.latest_costmap.info.origin.position.x
        origin_y = self.latest_costmap.info.origin.position.y

        rospy.loginfo(f"Taille de la carte : {width}x{height}, Résolution : {resolution}")

        # Convertir les données du costmap en tableau numpy
        data = np.array(self.latest_costmap.data).reshape((height, width))

        # Trouver les indices des zones non découvertes
        unknown_indices = np.argwhere(data == 100)  # Zones non découvertes (100)

        rospy.loginfo(f"Nombre de cellules non découvertes trouvées : {len(unknown_indices)}")

        if len(unknown_indices) == 0:
            rospy.loginfo("Aucune zone non découverte disponible.")
            return

        # Trier les zones non découvertes par distance au centre
        sorted_unknown_indices = self.sort_indices_by_distance_to_center(unknown_indices, width, height)

        for closest_index in sorted_unknown_indices:
            i, j = closest_index

            # Rechercher une zone libre autour de cette cellule
            target_cell = self.find_adjacent_free_cell(i, j, data)
            if target_cell is not None:
                target_i, target_j = target_cell
                x = origin_x + target_j * resolution + resolution / 2
                y = origin_y + target_i * resolution + resolution / 2

                rospy.loginfo(f"Coordonnées de la zone libre proche sélectionnée : x={x}, y={y}")
                self.send_goal(x, y)
                return

        rospy.logwarn("Aucune cellule libre trouvée pour atteindre une zone non découverte.")

    def sort_indices_by_distance_to_center(self, indices, width, height):
        """
        Trie les indices des zones inconnues par distance croissante par rapport au centre de la carte.
        """
        center_y, center_x = height // 2, width // 2
        distances = np.linalg.norm(indices - np.array([center_y, center_x]), axis=1)
        sorted_indices = indices[np.argsort(distances)]
        return sorted_indices

    def find_adjacent_free_cell(self, i, j, data):
        """
        Trouve une cellule libre adjacente à une cellule inconnue.
        """
        # Définir les déplacements pour les cellules adjacentes (8-connexité)
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                      (-1, -1), (-1, 1), (1, -1), (1, 1)]

        best_cell = None
        min_distance = float('inf')

        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < data.shape[0] and 0 <= nj < data.shape[1]:  # Vérifier les limites
                if data[ni, nj] == 0:  # Zone libre
                    distance = np.linalg.norm([di, dj])
                    if distance < min_distance:
                        min_distance = distance
                        best_cell = (ni, nj)

        return best_cell  # Retourne la meilleure cellule libre trouvée (ou None)

    def send_goal(self, x, y):
        """Crée et publie un objectif PoseStamped à move_base et attend que le robot atteigne la position"""
        self.is_moving = True
        goal = PoseStamped()
        goal.header.frame_id = "robot_map"
        goal.header.stamp = rospy.Time.now()

        goal.pose.position.x = x
        goal.pose.position.y = y
        goal.pose.orientation.w = 1

        rospy.loginfo(f"Envoi d'un objectif à x={x}, y={y}")
        self.goal_publisher.publish(goal)

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