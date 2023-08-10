import cv2
import mediapipe as mp

# Initialize MediaPipe Hand Tracking
mp_hands = mp.solutions.hands

# Constants for canvas dimensions
CANVAS_WIDTH, CANVAS_HEIGHT = 640, 480

# Create an AR scene with a virtual object (square)
virtual_object_size = 50
virtual_object_pos = (CANVAS_WIDTH // 2, CANVAS_HEIGHT // 2)

def move_object(x, y):
    global virtual_object_pos
    virtual_object_pos = (x, y)

def main():
    global virtual_object_pos

    cap = cv2.VideoCapture(0)
    cap.set(3, CANVAS_WIDTH)
    cap.set(4, CANVAS_HEIGHT)

    with mp_hands.Hands(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Failed to capture frame.")
                break

            # Convert the image to RGB for MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the hand landmarks using MediaPipe
            results = hands.process(image_rgb)

            # Draw the virtual object (square) on the AR scene
            cv2.rectangle(image, (virtual_object_pos[0] - virtual_object_size // 2,
                                  virtual_object_pos[1] - virtual_object_size // 2),
                          (virtual_object_pos[0] + virtual_object_size // 2,
                           virtual_object_pos[1] + virtual_object_size // 2),
                          (0, 255, 0), -1)

            # Move the virtual object based on hand landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Get the x, y coordinates of the index finger tip
                    index_finger_tip_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * CANVAS_WIDTH)
                    index_finger_tip_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * CANVAS_HEIGHT)

                    # Move the virtual object (square) to the position of the index finger tip
                    move_object(index_finger_tip_x, index_finger_tip_y)

            # Display the AR scene with the virtual object
            cv2.imshow('AR Scene', image)

            # Exit the loop when 'q' is pressed
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
