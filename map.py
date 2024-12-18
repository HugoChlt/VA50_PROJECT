#!/usr/bin/env python3

import rospy
import numpy as np
import cv2
from nav_msgs.msg import OccupancyGrid
from geometry_msgs.msg import PoseStamped

class UndiscoveredZoneExplorer:
    def __init__(self):
        """Initialise le noeud ROS, abonnez-vous au topic de la carte et configurez le publisher pour move_base."""
        rospy.init_node('undiscovered_zone_explorer', anonymous=True)
        
        # Paramètres de l'explorateur
        self.map_topic = rospy.get_param('~map_topic', '/robot/map')
        
        # Abonnement au topic costmap
        rospy.Subscriber(self.map_topic, OccupancyGrid, self.map_callback)
        
        # Stockage de la dernière carte reçue
        self.latest_map = None
        
        # Indique si le robot est en mouvement ou non
        self.is_moving = False

    def map_callback(self, msg):
        """Callback appelé à chaque réception de la carte du costmap"""
        self.latest_map = msg
        rospy.loginfo("Nouvelle costmap reçue, analyse des zones non découvertes...")

        # Récupérer les dimensions de la carte
        width = self.latest_map.info.width
        height = self.latest_map.info.height
        resolution = self.latest_map.info.resolution
        origin_x = self.latest_map.info.origin.position.x
        origin_y = self.latest_map.info.origin.position.y

        data = np.array(self.latest_map.data).reshape((height, width))

        # Créez une image OpenCV avec les différentes catégories
        image = np.zeros_like(data, dtype=np.uint8)
        # image[data == 0] = 255    # Zones libres (blanc)
        # image[data == 100] = 127  # Zones non découvertes (gris)
        # image[data == 99] = 0     # Murs (noir)

        # Enregistrer les données de la carte dans un fichier texte
        with open('/tmp/map_data.txt', 'w') as f:
            np.savetxt(f, data, fmt='%d')
        
        # Afficher la carte
        resized_image = cv2.resize(image, (800, 800), interpolation=cv2.INTER_NEAREST)
        cv2.imshow('Map', resized_image)
        cv2.waitKey(0)

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