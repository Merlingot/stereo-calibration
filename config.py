# Dossier où sont les images de calibration à analyser
calibration_folder = 'captures_calibration/'

# Path vers les images à analyser
single_path=calibration_folder # images individuelles
stereo_path=calibration_folder # images stéréo (peuvent être les mêmes que individuelles)

# Fichiers de calibration:
left_xml='cam1.xml'
right_xml='cam2.xml'

# Paramètres du damier
patternSize = (6,8)
squareSize = 3e-2

# Path vers les folders où enregistrer les images détectées
single_detected_path='output/singles_detected/' #images détectée lors de la calibration individuelle
stereo_detected_path='output/stereo_detected/'#images détectée lors de la calibration stéréo
