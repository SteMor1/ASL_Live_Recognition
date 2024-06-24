import cv2
import json
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# Carica il modello dal file
"""with open('./Modelli/tree.pkl', 'rb') as file:
    classification_model = pickle.load(file)"""

def apply_edge_detection_to_image(img):
    # Applicare il filtro di Sobel
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # Rileva bordi orizzontali
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # Rileva bordi verticali
    sobel_combined = cv2.magnitude(sobelx, sobely)
    # Normalizzare l'immagine Sobel combinata
    sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX)
    sobel_combined = np.uint8(sobel_combined)
    return sobel_combined
# Carica il modello di classificazione delle immagini
classification_model = load_model("./Modelli/nn_landmarks.h5")
with open("/labels/labels.json", 'r') as file_json:
    labels = json.load(file_json)
# Inizializza il rilevatore di mano di MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
# Funzione per rilevare le mani e classificare il gesto
def detect_and_classify_gesture(frame):
    # Converti il frame in scala di grigi
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Rileva le mani nel frame
    results = hands.process(rgb)
    image_height, image_width, _ = rgb.shape
    if results.multi_hand_landmarks:
        # Se una mano è stata rilevata, cattura l'immagine della mano
        for hand_landmarks in results.multi_hand_landmarks:
            # Cattura le coordinate dei landmark della mano
            hand_landmarks_np = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])
            # Converte le coordinate in un array 1D
            landmarks =[float(j) for a,i in enumerate(hand_landmarks_np) for j in i]
            landmarks = np.expand_dims(landmarks, axis=0)
            bbox = [min(hand_landmarks_np[:,0])*image_width, min(hand_landmarks_np[:,1])*image_height, max(hand_landmarks_np[:,0])*image_width, max(hand_landmarks_np[:,1])*image_height]
            print(landmarks)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            if len(hand_landmarks_np) > 0:
                # Ottieni le coordinate del rettangolo che circonda la mano
               
                
                # Disegna il rettangolo intorno alla mano sul frame
               
                #
                if(len(bbox)==4):
                    x = int(bbox[0])-50
                    y = int(bbox[1])-50
                    h = int(bbox[3]-bbox[1]+100)
                    w = int(bbox[2]-bbox[0]+100)
                    start=(x,y)
                    end=(x+w, y+h)
               
                    
                    hand_img = frame[y:y+h, x:x+w]
                    try:
                        hand_img_resized = cv2.resize(hand_img, (64,64))
                        
                        hand_img_resized = cv2.cvtColor(hand_img_resized, cv2.COLOR_BGR2GRAY)
                       
                        hand_img_norm = hand_img_resized/255
                        cv2.imshow('Hand Image', hand_img_resized)
                        hand_img_norm = np.expand_dims(hand_img_norm,axis=0)
                        gesture_class = classification_model.predict(landmarks)
                        #print(labels[np.argmax(gesture_class)])
                        cv2.rectangle(frame, start, end, (0, 255, 0), 2)
                        
                        cv2.putText(frame, labels[np.argmax(gesture_class)], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
                    except Exception as e:
                        print(e)

                # Visualizza il risultato della classificazione
                #print("Gesto classificato:", gesture_class)
                
                

            # Visualizza il risultato della classificazione
            #print("Gesto classificato:", gesture_class)

    # Ritorna il frame con il rettangolo intorno alla mano
    return frame

# Funzione principale
def main():
    # Apri il flusso video dalla telecamera
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        # Leggi il frame dal flusso video
        ret, frame = cap.read()

        if not ret:
            break

        # Rileva e classifica il gesto nel frame
        frame = detect_and_classify_gesture(frame)

        # Visualizza il frame
        cv2.imshow('Hand Gesture Recognition', frame)

        # Esci se viene premuto 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Rilascia le risorse
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
