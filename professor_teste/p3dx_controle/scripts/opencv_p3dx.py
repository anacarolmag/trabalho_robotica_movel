#! /usr/bin/env python3

import imutils
import cv2
import math

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
arucoParams = cv2.aruco.DetectorParameters_create()


class detection():
    def __init__(self):
        self.center = None
        self.markerID1 = None
        self.radius1 = None
        self.T = 0

        # Ordem de prioridade e tags já completadas
        self.priority_order = [1, 2, 3]
        self.completed_tags = []

    def aruco_detection(self, image):
        self.image = image
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = imutils.resize(image, width=1000)

        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

        if ids is not None and len(corners) > 0:
            ids = ids.flatten()

            # Seleciona a tag com maior prioridade
            selected_marker = None
            for prio_id in self.priority_order:
                if prio_id in ids and prio_id not in self.completed_tags:
                    idx = list(ids).index(prio_id)
                    selected_marker = (corners[idx], prio_id)
                    break

            if selected_marker:
                markerCorner, markerID = selected_marker
                pts = markerCorner.reshape((4, 2))
                (topLeft, topRight, bottomRight, bottomLeft) = pts

                # Conversão para inteiros
                topLeft = (int(topLeft[0]), int(topLeft[1]))
                topRight = (int(topRight[0]), int(topRight[1]))
                bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
                bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))

                # Centro = média dos 4 cantos
                cX = int((topLeft[0] + topRight[0] + bottomRight[0] + bottomLeft[0]) / 4.0)
                cY = int((topLeft[1] + topRight[1] + bottomRight[1] + bottomLeft[1]) / 4.0)

                # Tamanho aparente do marcador
                radius = int(math.sqrt(
                    (topRight[0] - bottomLeft[0]) ** 2 +
                    (topRight[1] - bottomLeft[1]) ** 2) / 2)

                # Desenho para visualização
                cv2.circle(image, (cX, cY), radius, (0, 0, 255), 3)
                cv2.putText(image, "Aruco Marker ID = " + str(markerID),
                            (topLeft[0] + 20, topLeft[1] - 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

                # Armazena resultado
                self.center = (cX, cY)
                self.markerID1 = markerID
                self.radius1 = radius

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            exit()

        return [gray, self.center, self.radius1, self.markerID1, image]
