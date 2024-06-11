import cv2
import mediapipe as mp
import numpy as np
from tensorflow.keras.models import load_model

# Carica il modello di classificazione delle immagini
classification_model = load_model("./Modelli/modello1")

# Inizializza il rilevatore di mano di MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)

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
            hand_image = np.expand_dims(hand_landmarks_np, axis=0)
            bbox = [min(hand_landmarks_np[:,0])*image_width, min(hand_landmarks_np[:,1])*image_height, max(hand_landmarks_np[:,0])*image_width, max(hand_landmarks_np[:,1])*image_height]
            # Classifica il gesto utilizzando il modello di classificazione
            #gesture_class = classification_model.predict(hand_image)
            
            # Verifica se ci sono punti validi
            if len(hand_landmarks_np) > 0:
                # Ottieni le coordinate del rettangolo che circonda la mano
               
                
                # Disegna il rettangolo intorno alla mano sul frame
               
                #
                start=(bbox[0], bbox[1])
                end=(bbox[2], bbox[3])
                cv2.rectangle(frame, start, end, (0, 255, 0), 2)

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
