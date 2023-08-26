# imports
import cv2
import mediapipe as mp
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def kneeBend(image, landmarks, params):
    params['name'] = 'Knee Bend'

    # Get coordinates
    left_hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
    left_knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    left_ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

    # Calculate angle
    angle = calculate_angle(left_hip, left_knee, left_ankle)
    params['angle'] = angle

    # displaying angle in the frame
    cv2.putText(image, str(int(angle)),
                tuple(np.multiply(left_knee, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA
                )

    ## Counter logic
    if (angle > 140 and (params['stage'] is None or params['stage'] == 'up')):
        if params['stage'] == 'up':
            # we store the time of the previous rep
            params['rep_time'] = abs(time.time() - params['t1'])

            # to avoid false positives
            if params['rep_time'] > 0.3 * params['threshtime']:
                params['rep_time_list'].append(params['rep_time'])
                params['counter'] += 1

        params['stage'] = 'down'
        params['t2'] = time.time()

    if angle > 140:
        params['t1'] = time.time()

    if angle < 140 and params['stage'] == 'down':
        params['t2'] = time.time()
        params['stage'] = 'up'

    return image, params


def add_box(image, color='r'):
    mp = {
        'g': (0, 255, 0),
        'r': (0, 0, 255),
        'b': (255, 0, 0)
    }

    cv2.rectangle(image, (0, 0), (640, 480), mp[color], 5)
    return image


def add_feedback(image, params):
    ## Adding rectangle for Reps
    cv2.rectangle(image, (0, 0), (int(0.15 * 640), 75), (135, 135, 88), -1)

    ## Adding rectangle for feedback
    cv2.rectangle(image, (int(0.15 * 640), 0), (640 - int(0.15 * 640), 75), (135, 135, 88), -1)

    ## Adding one more rectangle
    cv2.rectangle(image, (640 - int(0.15 * 640), 0), (640, 75), (135, 135, 88), -1)

    ## displaying reps
    cv2.putText(image, 'Reps', (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(image, str(params['counter']), (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    ## displaying Exercise Name
    cv2.putText(image, 'Exercise', (520, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.putText(image, str(params['name']), (520, 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

    ## Displaying timer
    if params['stage'] == 'up':
        cv2.putText(image, 'Hold Time', (100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, str(int(time.time() - params['t2'])), (120, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    else:
        cv2.putText(image, 'Relax Time', (100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.putText(image, str(int(time.time() - params['t2'])), (120, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    ## Display Feedback
    cv2.putText(image, 'Feedback', (250, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    if params['stage'] == 'down':
        # hold your knees bent if not holden knees more than 8 secs
        if (1.5 < params['rep_time'] < params['threshtime']):
            cv2.putText(image, 'Keep your knees bent', (250, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            image = add_box(image, 'r')

        # if user is not raising knees
        elif time.time() - params['t2'] > params['relaxtime']:
            cv2.putText(image, 'Raise your knees', (250, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            image = add_box(image, 'b')

    else:
        # if we hold knees more then 8secs we display relax and add green box
        if time.time() - params['t2'] > params['threshtime']:
            cv2.putText(image, 'Relax your knees', (250, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            image = add_box(image, 'g')

    return image


def check_time(params):
    colors = []
    for x in params['rep_time_list']:
        if x >= params['threshtime']:
            colors.append('green')
        elif x <= params['threshtime']:
            colors.append('red')

    return colors


def plot(params):
    ## getting color for different points
    col = check_time(params)

    plt.figure(figsize=(15, 4))
    plt.grid()
    plt.plot(range(1, params['counter'] + 1), params['rep_time_list'], 'y-.')

    for i in range(len(params['rep_time_list'])):
        plt.scatter(i + 1, params['rep_time_list'][i], c=col[i], s=70,
                    linewidth=0)

    plt.xticks(list(range(1, params['counter'] + 1)))
    plt.xlabel('Reps')
    plt.ylabel('Rep time')
    plt.title(params['name'])

    plt.text(5, max(params['rep_time_list']) - 1, 'Good Rep',
             fontsize=13, bbox=dict(facecolor='green', alpha=0.5))

    plt.text(10, max(params['rep_time_list']) - 0.5, 'Incomplete Rep',
             fontsize=13, bbox=dict(facecolor='red', alpha=0.5))

    try:
        plt.text(params['counter'] - 4, max(params['rep_time_list']) - 1, 'Max Angle: ' + str(int(params['max_angle'])),
                 fontsize=13, bbox=dict(facecolor='yellow', alpha=0.5))

        plt.text(params['counter'] - 2.5, max(params['rep_time_list']) - 1,
                 'Min Angle: ' + str(int(params['min_angle'])),
                 fontsize=13, bbox=dict(facecolor='yellow', alpha=0.5))
    except:
        pass

    plt.show()


if __name__ == '__main__':
    cap = cv2.VideoCapture('KneeBendVideo.mp4')

    ## define codec to save video
    fourcc = cv2.VideoWriter_fourcc('P', 'I', 'M', '1')
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('output.avi', fourcc, fps, (640, 480))

    params = {
        'counter': 0,
        'stage': None,
        't1': time.time(),
        't2': time.time(),
        'rep_time': 8,
        'name': None,
        'rep_time_list': [],
        'min_blur': 110,
        'angle': None,
        'threshtime': 8,
        'relaxtime': 4
    }

    ret, prev_img = cap.read()
    prev_img = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)

    ## Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5, model_complexity=1) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            if (not ret):
                break

            # if there is a fluctutation we skip current frame
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mask = cv2.absdiff(prev_img, img)
            prev_img = img

            if np.sum(mask) > 0.2 * 1e7:
                continue

            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (640, 480))
            image.flags.writeable = False

            # Make detection
            results = pose.process(image)

            # Recolor back to BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # if blurred frame we pass the frame
            # blur = cv2.Laplacian(image, cv2.CV_64F).var()

            #         if blur > params['min_blur']:
            #             continue

            # Extract landmarks
            try:
                landmarks = results.pose_landmarks.landmark
                image, params = kneeBend(image, landmarks, params)
                image = add_feedback(image, params)
            except:
                pass

            # Render detections
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=1),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=1)
                                      )

            out.write(image)

            cv2.imshow('Mediapipe Feed', image)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    plot(params)
