# Real-time-pos-eval Project

## Objective

Estimating pose similarity between a live video input and a reference video of high movement and complex data. Our system is focused on minimizing the effects of jitter, occlusion, and limb length variation between frames. We intend to make a low-latency, accurate processing pipeline that can account for varying body proportions to decrease bias.

## Analysis flow

- Step 1: load ground truth keypoints and raw video from AIST++ dataset into a dictionary and VideoCapture object, respectively
- Step 2: visualize ground truth keypoints and limb lines on top of each frame and save the resulting images back into a video file
- Step 3: Process raw video through mediapipe pose estimation model and save predicted keypoints of each frame to pkl file, making sure to map correctly to the same keypoints in the ground truth pkl file
- Step 4: create means of evaluation between the 2 sets of keypoints and of the quality of the predictions: MSE, Mean Jerk
- Step 5: visualize predicted keypoints compared to ground truth keypoints for AIST++ videos
- Step 6: process live video of people of varying sizes and run through mediapipe model
- Step 7: visualize live video mediapipe predicted keypoints compared to reference video mediapipe predicted keypoints.
- Step 8: Write post processing normalizaion for limb lengths and body proportions and compare performance with vanilla mediapipe
- Step 8: Implement optical flow and visualize only on AIST++ video
- Step 8: Implement kalmanFilter and kalmanNet only on AIST++ video and compare 
- Step 9: Implement kalmanNet with optical flow integration only on AIST++ video and compare to vanilla kalmanNet


## Tools

- Video parser for static and live video that generates the current frame and its specs
- Visualizer that edits the raw video to draw the keypoints and limb lines
- MediaPipePose object that processes a frame into a pkl file of keypoints corresponding to the same keypoints in the AIST++ dataset
- Evaluation object that takes in 2 sets of keypoints and returns the metrics that determine model performance
- limb length and body proportion normalization object that takes in 2 sets of keypoints and normalizes for comparison
- kalman filter object that can take in keypoints and return a temporally smoothed set of keypoints
- kalmanNet object that can take in keypoints and return a temporally smoothed set of keypoints
- optical flow object that takes in whole frame input and generates an velocity estimation for every point in the frame
- visualizer for the optical flow field of every frame, shows velocity arrows
- special kalmanNet that integrates optical flow velocity estimates into the correction process

