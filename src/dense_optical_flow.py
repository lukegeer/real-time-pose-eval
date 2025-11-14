import cv2
import numpy as np

# https://github.com/iv4n-ga6l/Optical_flow_opencv/blob/main/algorithms/dense_optical_flow.py

def dense_optical_flow(method, video_path, params=[], to_gray=False):
    # read the video
    cap = cv2.VideoCapture(video_path)
    # Read the first frame
    ret, old_frame = cap.read()

    # crate HSV & make Value a constant
    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # Preprocessing for exact method
    if to_gray:
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    while True:
        # Read the next frame
        ret, new_frame = cap.read()
        frame_copy = new_frame
        if not ret:
            break
        # Preprocessing for exact method
        if to_gray:
            new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        # Calculate Optical Flow
        flow = method(old_frame, new_frame, None, *params)

        # Encoding: convert the algorithm's output into Polar coordinates
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        # Use Hue and Saturation to encode the Optical Flow
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # Convert HSV image into BGR for demo
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
        cv2.imshow("frame", frame_copy)
        cv2.namedWindow("optical flow", cv2.WINDOW_NORMAL)
        cv2.imshow("optical flow", bgr)
        k = cv2.waitKey(25) & 0xFF
        if k == 27:
            break
        old_frame = new_frame

def main():
    video_path = "data/videos/gBR_sBM_c01_d04_mBR0_ch01.mp4"
    # print(os.path.exists(video_path))

    # methods
    method = cv2.calcOpticalFlowFarneback
    params = [0.5, 3, 15, 3, 5, 1.2, 0]
    to_gray = True

    # method = cv2.optflow.DualTVL1OpticalFlow_create().calc
    # params = []
    # to_gray=True

    dense_optical_flow(method, video_path, params, to_gray)

if __name__ == "__main__":
    main()
