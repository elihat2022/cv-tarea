import cv2 as cv
from pathlib import Path

backSub = cv.createBackgroundSubtractorMOG2(history=40,varThreshold=40, detectShadows=True)
ruta_video = Path("video.mov")
carpeta_salida = Path("fotogramas_con_alfa")
carpeta_salida.mkdir(exist_ok=True)


kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (8, 8))
capture = cv.VideoCapture(str(ruta_video))

fotogramas_fondo = 120
numero_frame = 0

try:
    while True:
        ret, frame = capture.read()
        if frame is None:
            break
        
        numero_frame += 1
        blur = cv.GaussianBlur(frame, (5, 5), 0)
        # Aplicar el modelo de fondo para obtener la máscara de primer plano
        if numero_frame <= fotogramas_fondo:
                
                backSub.apply(blur, learningRate=-1)
                cv.imshow("Frame", frame)

        #actualizar el modelo del fondo
        else:
            fgMask = backSub.apply(blur, learningRate=0)
            _, fgMask = cv.threshold(fgMask, 254, 255, cv.THRESH_BINARY)

            # Eliminar pequeños puntos y cerrar huecos pequeños.
            fgMask = cv.morphologyEx(fgMask, cv.MORPH_OPEN, kernel)
            fgMask = cv.morphologyEx(fgMask, cv.MORPH_CLOSE, kernel)

            fotograma_bgra = cv.cvtColor(frame, cv.COLOR_BGR2BGRA)
            fotograma_bgra[:, :, 3] = fgMask

            primerPlano = cv.bitwise_and(frame, frame, mask=fgMask)

            cv.imshow("Frame", frame)
            cv.imshow("FG Mask", fgMask)
            cv.imshow("Primer plano", primerPlano)

            # Guardar el fotograma con alfa
            nombre = carpeta_salida / f"frame_{numero_frame:04d}.png"
            cv.imwrite(str(nombre), fotograma_bgra)

        keyboard = cv.waitKey(30) & 0xFF
        if keyboard == ord("q") or keyboard == 27:
            break
finally:
    capture.release()
    cv.destroyAllWindows()