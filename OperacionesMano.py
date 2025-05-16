import cv2
import mediapipe as mp
import time

class detectormanos:
    def __init__(self, mode=False, maxManos=1, Confdeteccion=0.75, Confsegui=0.5):
        self.mode = mode
        self.maxManos = maxManos
        self.Confdeteccion = Confdeteccion
        self.Confsegui = Confsegui

        self.mpmanos = mp.solutions.hands
        self.manos = self.mpmanos.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxManos,
            min_detection_confidence=self.Confdeteccion,
            min_tracking_confidence=self.Confsegui
        )
        self.dibujo = mp.solutions.drawing_utils
        self.tip = [4, 8, 12, 16, 20]

    def encontarmanos(self, frame, dibujar=True):
        imgcolor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.resultados = self.manos.process(imgcolor)

        if self.resultados.multi_hand_landmarks:
            for mano in self.resultados.multi_hand_landmarks:
                if dibujar:
                    self.dibujo.draw_landmarks(frame, mano, self.mpmanos.HAND_CONNECTIONS)
        return frame

    def encontrarposicion(self, frame, ManoNum=0, dibujar=True):
        xlista = []
        ylista = []
        bbox = []
        self.lista = []
        if self.resultados.multi_hand_landmarks:
            miMano = self.resultados.multi_hand_landmarks[ManoNum]
            for id, lm in enumerate(miMano.landmark):
                alto, ancho, _ = frame.shape
                cx, cy = int(lm.x * ancho), int(lm.y * alto)
                xlista.append(cx)
                ylista.append(cy)
                self.lista.append([id, cx, cy])
                if dibujar:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 0), cv2.FILLED)

            if xlista and ylista:
                xmin, xmax = min(xlista), max(xlista)
                ymin, ymax = min(ylista), max(ylista)
                bbox = xmin, ymin, xmax, ymax
                if dibujar:
                    cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
        return self.lista, bbox

    def dedosarriba(self):
        dedos = []
        if not self.lista:
            return dedos

        # Pulgar
        if self.lista[self.tip[0]][1] > self.lista[self.tip[0] - 1][1]:
            dedos.append(1)
        else:
            dedos.append(0)

        # Otros dedos
        for id in range(1, 5):
            if self.lista[self.tip[id]][2] < self.lista[self.tip[id] - 2][2]:
                dedos.append(1)
            else:
                dedos.append(0)

        return dedos

# Código principal
detector = detectormanos(Confdeteccion=0.75)
cap = cv2.VideoCapture(0)

d1 = 0
d2 = 0
r = 0
cx = 165
cy = 420
comp = 0
res = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    t = cv2.waitKey(1)
    frame = detector.encontarmanos(frame)
    manosInfo, cuadro = detector.encontrarposicion(frame, dibujar=False)

    cv2.putText(frame, "Dedos", (52, 145), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    cv2.putText(frame, "Digito 1", (45, 445), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    cv2.putText(frame, str(d1), (65, 420), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 10)
    cv2.putText(frame, "Digito 2", (245, 445), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    cv2.putText(frame, str(d2), (265, 420), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 10)
    cv2.putText(frame, "Resultado", (465, 445), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    cv2.putText(frame, str(r), (495, 420), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 10)

    if len(manosInfo) != 0:
        dedos = detector.dedosarriba()
        contar = dedos.count(1)
        cv2.putText(frame, str(contar), (65, 125), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 10)

        # Suma
        if t in [83, 115]:  # Tecla S o s
            comp = 1
            d1 = contar
            res = 0
        if comp == 1:
            cv2.putText(frame, "+", (cx, cy), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 10)

        # Resta
        if t in [82, 114]:  # Tecla R o r
            comp = 2
            d1 = contar
            res = 0
        if comp == 2:
            cv2.putText(frame, "-", (cx, cy), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 10)

        # Multiplicación
        if t in [77, 109]:  # Tecla M o m
            comp = 3
            d1 = contar
            res = 0
        if comp == 3:
            cv2.putText(frame, "x", (cx, cy), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 10)

        # División
        if t in [68, 100]:  # Tecla D o d
            comp = 4
            d1 = contar
            res = 0
        if comp == 4:
            cv2.putText(frame, "/", (cx, cy), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 10)

        # Mostrar resultado
        if t == 32:  # Tecla espacio
            d2 = contar
            res = 1

        if res == 1:
            cv2.putText(frame, str(d2), (265, 420), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 10)
            cv2.putText(frame, "=", (365, cy), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 10)
            if comp == 1:
                r = d1 + d2
            elif comp == 2:
                r = d1 - d2
            elif comp == 3:
                r = d1 * d2
            elif comp == 4:
                r = round(d1 / d2, 2) if d2 != 0 else "Err"

        # Reset
        if t in [67, 99]:  # Tecla C o c
            d1 = d2 = r = comp = res = 0

    cv2.imshow("Manitas Matematicas", frame)
    if t == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
