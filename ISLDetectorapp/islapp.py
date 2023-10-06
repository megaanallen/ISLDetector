import pickle
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Load the model from the pickled file
model_dict = pickle.load(open('model.p', 'rb'))
model = model_dict['model']

# Initialize mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Initialize the labels dictionary
labels_dict = {0: 'A', 1: 'B', 2: 'C'}

def main():
    st.title("Hand Gesture Recognition App")
    st.write("This app uses a hand gesture recognition model to predict letters based on hand movements.")

    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    with st.spinner('Loading camera...'):
        if not cap.isOpened():
            st.error("Error: Camera not found.")
            return

        while True:
            # Read the frame
            ret, frame = cap.read()

            if not ret:
                st.error("Error: Frame is None or camera not found.")
                break

            # Flip the frame horizontally for a later selfie-view display
            H, W, _ = frame.shape

            # Convert the frame from BGR color (which OpenCV uses) to RGB color
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame and find the landmarks
            results = hands.process(frame_rgb)

            # Initialize the data list
            data_aux = []
            x_ = []
            y_ = []

            # If the landmarks are detected and the number of hands in the frame is 1
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) == 1:
                # For the hand
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw the hand landmarks on the frame
                    mp_drawing.draw_landmarks(
                        frame,  # image to draw
                        hand_landmarks,  # model output
                        mp_hands.HAND_CONNECTIONS,  # hand connections
                        mp_drawing_styles.get_default_hand_landmarks_style(),  # hand landmarks style
                        mp_drawing_styles.get_default_hand_connections_style())  # hand connections style

                # For the hand landmarks in the frame (only one hand)
                for hand_landmarks in results.multi_hand_landmarks:
                    # For each landmark in the hand (21 landmarks)
                    for i in range(len(hand_landmarks.landmark)):
                        # Get the x and y coordinates of the landmark
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y

                        # Append the coordinates to the x_ and y_ lists
                        x_.append(x)
                        y_.append(y)

                    # For each landmark in the hand (21 landmarks)
                    for i in range(len(hand_landmarks.landmark)):
                        # Get the x and y coordinates of the landmark
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        # Append the normalized coordinates to the data list
                        data_aux.append(x - min(x_))
                        data_aux.append(y - min(y_))

                # Get the bounding box coordinates of the hand
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10

                # Get the bounding box coordinates of the hand
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Get the prediction
                prediction = model.predict([np.asarray(data_aux)])

                # Get the predicted character
                predicted_character = labels_dict[int(prediction[0])]

                # Draw the green bounding box and the predicted yellow character
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 2)
                cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3,
                            cv2.LINE_AA)

                st.image(frame, channels="BGR", use_column_width=True)

if __name__ == '__main__':
    main()
