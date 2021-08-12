import cv2 as cv
from modules.util import clean_folders

from config import single_path as folder

################################################################################
# Déclaration des périphériques de capture
cap_left = cv.VideoCapture(0)
cap_right = cv.VideoCapture(1)
################################################################################


# Variables textuel
text_title="Assistant de calibration"
text_progress=""
# Nombres de captures prises
capture_taken=0
# Nombres de captures à prendre et folder
capture_qty=40
# Création du folder
clean_folders([folder])


while(True):
    if capture_taken==0:
        text_progress="Appuyez sur 'C' pour capturer les images"
    elif capture_taken==capture_qty:
        text_progress="Termine, maintenir 'Esc' pour quitter."
    else:
        text_progress="Il reste " + str(capture_qty-capture_taken)+ " image(s) a prendre."

    # Acquisition du flux video
    _, left_frame_color = cap_left.read()
    _, right_frame_color = cap_right.read()


    #Transformation de l'image en nuances de gris
    left_frame = cv.cvtColor(left_frame_color, cv.COLOR_BGR2GRAY)
    right_frame = cv.cvtColor(right_frame_color, cv.COLOR_BGR2GRAY)

    # Enregistrement des deux frames sans écritures
    if cv.waitKey(1) == ord('c'):
        if capture_taken<capture_qty:
            capture_taken = capture_taken + 1
            text_progress=""
            filename_l="{}left{:03d}".format(folder,capture_taken) + ".jpg"
            filename_r="{}right{:03d}".format(folder,capture_taken) + ".jpg"
            cv.imwrite(filename_l, left_frame)
            cv.imwrite(filename_r, right_frame)

    # Affichage du contenu
    left_view = left_frame_color #cv.resize(left_frame_color,(480,270))
    right_view = right_frame_color #cv.resize(right_frame_color,(480,270))


    left_view=cv.flip(left_view,1)
    right_view=cv.flip(right_view,1)


    cv.putText(left_view, text_progress, (60, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv.putText(right_view, text_progress, (60, 60), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv.imshow(text_title + ' Camera Gauche',left_view)
    cv.imshow(text_title + ' Camera Droite',right_view)

    if cv.waitKey(1) == 27:
        break

# On relache la connexion
cap_left.release()
cap_right.release()
cv.destroyAllWindows()
