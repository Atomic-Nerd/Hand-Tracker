import mediapipe as mp
import cv2
import time
import pyttsx3
import threading

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

def say(value):
    engine = pyttsx3.init()
    engine.say(str(value))
    engine.runAndWait()

def binary_to_decimal(binary_list):
    return sum(val * (2 ** idx) for idx, val in enumerate(reversed(binary_list)))

def ccw(A, B, C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


def lines_intersect(A, B, C, D):
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

if __name__ == "__main__":
    print ("Beginning video capture. Press 'q' to quit.")
    with mp_hands.Hands(
        max_num_hands=2, 
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
        ) as hands:

        #binaryStates = [0,0,0,0,0,0,0,0]
        while True:
            success, img = cap.read() 
            attempt = 0
            while not success and attempt < 5:
                time.sleep(0.2)  # Wait 0.2 seconds before retrying
                success, img = cap.read()
                attempt += 1
            if not success:
                print("Failed to capture image after 5 attempts.")
                break

            img = cv2.flip(img, 1)
            h, w = img.shape[:2]
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(rgb)

            #past_binaryStates = binaryStates.copy()  # Create a copy of the current state to compare later
            index_coords = {
                "Left" : None,
                "Right" : None
            }
            if results.multi_hand_landmarks:
                for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    hand_label = results.multi_handedness[hand_index].classification[0].label
                    mp_draw.draw_landmarks(
                        img, 
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS
                        )

                    finger_tips = {
                        "Thumb": hand_landmarks.landmark[4],
                        "Index": hand_landmarks.landmark[8],
                        "Middle": hand_landmarks.landmark[12],
                        "Ring": hand_landmarks.landmark[16],
                        "Pinky": hand_landmarks.landmark[20]
                    }


                    '''
                    if hand_label == "Right":
                        binaryStates[4] = 1 if finger_tips["Index"].y < hand_landmarks.landmark[6].y else 0
                        binaryStates[5] = 1 if finger_tips["Middle"].y < hand_landmarks.landmark[10].y else 0
                        binaryStates[6] = 1 if finger_tips["Ring"].y < hand_landmarks.landmark[14].y else 0
                        binaryStates[7] = 1 if finger_tips["Pinky"].y < hand_landmarks.landmark[18].y else 0
                    else:
                        binaryStates[3] = 1 if finger_tips["Index"].y < hand_landmarks.landmark[6].y else 0
                        binaryStates[2] = 1 if finger_tips["Middle"].y < hand_landmarks.landmark[10].y else 0
                        binaryStates[1] = 1 if finger_tips["Ring"].y < hand_landmarks.landmark[14].y else 0
                        binaryStates[0] = 1 if finger_tips["Pinky"].y < hand_landmarks.landmark[18].y else 0
                    '''
                    finger_coords = {}
                    for finger, tip in finger_tips.items():
                        x, y = int(tip.x * w), int(tip.y * h)

                        finger_coords[finger] = (x, y)

                        cv2.circle(img, (x, y), 5, (0, 255, 0), cv2.FILLED)
                        cv2.putText(img, f"{hand_label[0]}: {finger}", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

                    index_coords[hand_label] = finger_coords
                    

            '''
            if past_binaryStates != binaryStates:
                value = binary_to_decimal(binaryStates)
                print(
                    f"Finger States Changed: {binaryStates} -> Decimal Value: {value}"
                )

                threading.Thread(
                    target=say,
                    args=(value,),
                    daemon=True
                ).start()
            '''

            if index_coords["Left"] and index_coords["Right"]:
                #Line between Indexs
                cv2.line(img, index_coords["Left"]["Index"], index_coords["Right"]["Index"], (255, 0, 0), 2)

                #Line between Thumbs
                cv2.line(img, index_coords["Left"]["Thumb"], index_coords["Right"]["Thumb"], (0, 255, 0), 2)

                #Line between Index and Thumb Left
                cv2.line(img, index_coords["Left"]["Index"], index_coords["Left"]["Thumb"], (0, 255, 0), 2)


                #Line between Index and Thumb Right
                cv2.line(img, index_coords["Right"]["Index"], index_coords["Right"]["Thumb"], (0, 255, 0), 2)

                if lines_intersect(index_coords["Left"]["Index"], index_coords["Right"]["Index"], index_coords["Left"]["Thumb"], index_coords["Right"]["Thumb"]):
                    cv2.putText(img, "Hands Intersecting!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print("Hands Intersecting!")

            cv2.imshow("Image", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()