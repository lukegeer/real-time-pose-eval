# Evaluating the Limits of Real-Time Complex Pose Estimation and Reference Matching
Maaz Shamim - mshamim2 | Max-Peter Schrøder - mschrod2 | Lukas Geer - lgeer3  | Valerie Liang - vliang5 

## Introduction

Real-time human pose estimation has become increasingly important in interactive applications such as motion-based gaming, fitness monitoring, and sports analysis. These systems typically track a human body using a sparse set of keypoints representing joints to form a geometric skeleton. Because this representation is compact and robust to appearance variations, it enables fast inference even on lightweight devices, making it well suited for live webcam-based interaction.
Classic systems such as OpenPose and newer models like BlazePose, MoveNet, and YOLO-Pose have significantly advanced real-time pose detection by optimizing convolutional neural networks for speed and spatial precision. Despite these improvements, state-of-the-art methods still struggle with challenges including self-occlusion, external occlusion, and motion blur.
Recent work has attempted to address such issues through temporal modeling using optical flow, recurrent networks, temporal convolution, or attention mechanisms to infer obscured keypoints using past frames. Models such as HRNet-Flow, TCN-based pose refiners [6], and transformer-based trackers such as TokenPose [7] have demonstrated that temporal cues can significantly reduce jitter and recover occluded limbs. However, these methods are either computationally expensive, unsuitable for low-latency deployment, or still degrade under rapid motion.
This project evaluates the practical limits of real-time pose estimation in scenarios that require precise alignment between a user’s live movement and a reference demonstration. We compare several modern real-time models under challenging conditions such as occlusion, fast motion, and jitter. To improve stability, we develop custom refinement methods using Kalman filtering and optical-flow propagation to recover missing keypoints and smooth noisy predictions. We then evaluate multiple pose-similarity measures and assess their effectiveness in determining how closely a user matches a reference motion, with relevance to applications such as dance games and sports training. Our experiments focus on a curated subset of full-body movements where accuracy and temporal consistency are essential, allowing us to clearly characterize both the strengths and limitations of current real-time pose estimation methods.

## Data

To evaluate and train our algorithms we will use the following datasets:
1. AIST++: A large-scale dance dataset with 2D/3D annotations. This provides the complex, dynamic poses necessary to rigorously test our system against challenges like self-occlusion and rapid movement.
2. Martial Art, Dancing, Sports Dataset (MADS): A challenging motion dataset containing dance and martial arts with frequent occlusion and motion blur, used to test pose estimation under fast, highly dynamic movements.
3. Custom Dataset: A synthetic occlusion test set derived from AIST++ by adding 25×25 colored square patches centered on joints, sampled using a binomial distribution to simulate realistic body-to-body and object occlusions.

## Methods

Our system is built around two main components: keypoint detection and pose similarity measurement. For keypoint detection, we compare several strategies, including off-the-shelf pose estimators, temporal smoothing using Kalman Filter, optical-flow tracking, and a fine-tuned YOLO model. For pose similarity, we evaluate multiple scoring methods ranging from simple distance metrics to embedding-based comparisons. These approaches address different challenges in real-time pose alignment, and the following sections describe each method in detail.

### Keypoint Detection

To understand the strengths and limitations of existing pose estimation models, we first evaluated several off-the-shelf methods on AIST++ and MADS dataset. Table 1 shows per-joint performance across multiple models using metrics such as PJE (per-joint error) and PCK@20%, while Table 2 summarizes overall model performance in terms of MPJPE, PCK@20%, and the number of frames evaluated.

While some models, such as ViTPose, achieve very high accuracy in keypoint detection, their computational requirements make them impractical for real-time applications. In contrast, lightweight models like MediaPipe and YOLO11 offer fast inference suitable for live deployment, but they still suffer from challenges identified earlier, including temporal jitter, occlusion, and occasional missing keypoints.


#### Leveraging Optical Flow

This system addresses a common challenge in pose estimation: when body keypoints become occluded or are detected with low confidence, traditional pose estimators struggle to track them. By leveraging dense optical flow, we can predict the position of missing keypoints based on motion patterns from previous frames.
Dense optical flow (specifically using the Farneback method) computes motion vectors for every pixel between consecutive frames. Since dense optical flow is computationally intensive, we implemented optimizations including resolution reduction and frame skipping to maintain real-time performance. This represents a typical performance-robustness trade-off common in real-time computer vision applications.
The system processes video at two different resolutions to balance accuracy and speed: Original Resolution (1920×1080): MediaPipe pose estimation runs on full-resolution frames to maximize keypoint detection accuracy, and Downscaled Resolution (480×270): Optical flow computation runs on reduced resolution (downscaled 4x) to enable low-latency real-time performance
MediaPipe detects 33 body keypoints with associated confidence scores. When keypoints are clear and visible, MediaPipe provides accurate localization. However, when body parts are: occluded by other body parts or objects, out of frame boundaries, or moving at high velocities, confidence scores drop below reliable thresholds, and detections may become inaccurate or missing entirely. The primary challenges observed in the AIST dataset are occlusion and rapid motion. When a keypoint's confidence falls below the threshold (default: 0.5), the system predicts its new position using optical flow.

Prediction Procedure:

1. Retrieve the keypoint's previous position in original resolution
2. Map the keypoint location to corresponding coordinates in the flow field
3. Extract the flow vector at that location
4. Scale the flow vector back to original resolution
5. Predict the new keypoint position by applying the scaled flow vector

We implemented temporal smoothing to improve consistency in keypoint predictions. The system maintains information from the last N frames (default: 5) and applies temporal averaging. This smoothing is exclusively applied to keypoints predicted by optical flow, not to MediaPipe's direct detections.

#### Fine Tuning YOLOv11

To improve YOLOv11’s robustness to occluded joints, we generated a custom training dataset from the AIST++ pose dataset. For each frame, we randomly selected a subset of joints and applied small occlusion patches of size 25×25 pixels to simulate missing or partially hidden joints. This process was applied probabilistically: each frame had a 40% chance of being occluded, and within an occluded frame, each joint had a 30% chance of being covered. The occluded images were saved along with corresponding YOLO-formatted labels indicating the visible joints, enabling the model to learn to predict key points even when some joints are temporarily missing. This augmentation strategy allows YOLOv11 to handle real-world occlusions while maintaining its real-time inference capabilities.

#### Kalman Filter

Implemented as a post-processing technique to smooth joint trajectories and reduce jitter. Models not just position, but also velocity and acceleration to account for high-speed movements. 
The Kalman Filter is a recursive algorithm used to estimate the state of a dynamic system from noisy observations. In the context of pose estimation, it smooths keypoint trajectories over time by balancing predictions based on motion dynamics with new measurements from a pose detector. The algorithm comprises of two main steps: 
Predict Step: Using the previous estimated state (e.g., position and velocity), the Kalman Filter predicts the system's next state assuming a linear motion model. It also updates the estimated uncertainty. 

Update Step: When a new observation (noisy keypoint position) becomes available, the Kalman Filter adjusts its prediction by computing a weighted average between the predicted state and the new measurement. The weighting is determined by their respective uncertainties: more uncertain measurements are given less influence.

This iterative process enables the Kalman Filter to produce smoothed, temporally consistent estimates of keypoint locations, reducing jitter and helping maintain stability even in frames with missing or unreliable detections.

#### Kalman Filter with Optical Flow and Acceleration

Traditionally, the kalman filter assumes a constant velocity and does not include acceleration in the state vector. However, we can include both of these as a measurement in the systems to add more information which we gain with optical flow. With the proposed benefits of optical flow being able to track keypoints accurately through occlusion, we can assume optical flow to be an accurate estimation of velocity and use it for the velocity measurement and compute the acceleration using the velocity from the current and previous frame. 
Integrating velocity and acceleration measurement requires adding dimensions to the F, H, Q, and R matrices to expand them. The F matrix incorporates velocity and acceleration in the state transition, the H in the control input. The Q and R matrices control smoothing and the magnitude of inclusion of acceleration and velocity in the smoothing update. We chose to smooth more heavily with the velocity and acceleration terms rather than the position terms. 
We proposed that including optical flow as velocity and acceleration in our kalman filter allows for increased prediction power as to the next position of each keypoint without reliance on the low confidence occluded location predictions and measurements.

### Pose Similarity Measurements:

Once keypoints are detected, we evaluate different methods to quantify how closely a user’s pose matches a reference pose. We focus on two complementary approaches:

1. Pose Pairwise Distance Similarity Error:  Rotation and scale invariant evaluation similarity metric that takes the pairwise distance between every joint keypoint and every other joint keypoint on the body, these pairwise distances are then normalized by the sum of all pairwise distances on the skeleton. This process is done independently for each. After normalization, the normalized difference in corresponding pairwise distances are computed and the average difference is taken and used as the error metric. The nature of pairwise distance being relative to other keypoints on the body rather than just between each individual corresponding keypoint gives it rotation invariance because the information lost to rotation is not included in the distance between keypoints, e.g. if someone rotates in 2d, the distance between a keypoint and another does not change. It is scale invariant due to the normalization of the keypoints which eliminates any differences in scale. Since the error metric doesn’t map to a scalar between 0 and 1 for our match score, we have to apply an additional transformation.
2. Pose Embedding NN: A network that learns the underlying pose embeddings that maximizes the cosine similarity between embeddings that map to the same pose. The pose embedding network is trained on a triplet dataset derived from AIST++ 2D keypoints, composed of positive and negative keypoint pairs: positive matches represent similar poses and negative matches represent dissimilar poses. This sampling strategy ensures the network learns to distinguish between pose similarities and differences. Our loss is defined as the contrastive cosine similarity between the resulting pose embedding outputs from the neural network. It is contrastive as we include a positive term for the assumed positive matches and a negative term that instates a margin to clamp the loss to zero. Both terms are squared to penalize smaller scaled similarities more.

## Evaluation

In this section, we evaluate the performance of our keypoint detection and pose similarity methods across multiple datasets and conditions.

### Optical Flow

This experiment evaluates how often the optical flow-based fallback system is required to compensate for unreliable or missing pose keypoints in the AIST dance dataset. The results indicate that optical flow is required intermittently, approximately 5 to 18% of the time across all videos. All channels exhibit relatively high standard deviations (0.55–0.92), reflecting bursts of optical flow employment  rather than continuous reliance on optical flow throughout the video. In the most severe cases, up to 27 keypoints are replaced by optical flow estimates. This variability is expected in dynamic human motion and confirms that the fallback system activates exactly when detection uncertainty spikes.
In most frames, the SOTA detector succeeds, but during temporary failures (such as rapid spins or heavy occlusions) the system may need to predict a substantial number of keypoints. This shows that the fallback system is engaged regularly but not excessively, meaning the optical flow component is behaving as intended. This pattern is consistent with the AIST dataset: while MediaPipe performs well on many frames, dancers frequently produce poses that temporarily hide limbs or create motion blur, causing confidence scores to drop below the threshold.
Overall, the results validate the motivation for optical flow-based keypoint estimation. Optical flow meaningfully improves temporal consistency and helps preserve continuous tracking during challenging motion, without overwhelming the detector or introducing unnecessary corrections.

### Fine-Tuned YOLOv11

Due to limited computational capacity, we fine-tuned the YOLOv11 pose model on a small subset of AIST++ 30 videos (each 15 seconds) sampled at 60 fps. The model was trained for 20 epochs. With this restricted training setup, the fine-tuned model achieved an improvement of approximately 1.5% over the original YOLOv11 checkpoint. 

### Kalman Filter

We designed our evaluation not only on the accuracy of the model compared to the ground truth, but on the similarity between the predictions for each corresponding frame of an assumed matching pair of videos of different dancers doing the same choreography. Our prediction pipeline techniques force us to evaluate in this way due to their smoothing effect, causing a lag in the predictions. We can only evaluate the accuracy of the model’s raw predictions with the ground truth. So we choose to evaluate by comparing the two input videos after running both through the chosen processing pipeline and computing the similarity of the predicted keypoints for each frame. Evaluating in this way is valid because we intended to create a product that computes the degree of matching from one frame to another, so an exact match to the ground truth keypoints is not necessary as we compare the predicted key points for both the reference and live video. Additionally, to generate a robust evaluation, we created several rotation and scale invariant metrics.

#### Performance on Similarity Metrics (AIST++)

We evaluated each model, base MediaPipe, MediaPipe with classical kalman, and MediaPipe with kalman with optical flow and acceleration. Each model was evaluated on a random batch of 50 matched and 50 mismatched videos from the AIST++ dataset. Note: the pairwise distance error in this table is not transformed to scalar values between 0 and 1, but still reflects the improvement in similarity.

For the entire evaluation set, the MediaPipe with the kalman filter with optical flow and acceleration as measurements improve on the base MediaPipe model. However, this improvement is somewhat arbitrary as increasing the difference between the similarity metrics of the matches and mismatches within each model is our goal. Ideally, we want to create as separable of a threshold between matching and mismatched frames as possible. We only see a marginal increase in the difference between the embedding similarity in the MediaPipe w/ Kalman + Opt. Flow and Accel. This lack of a significant finding for this evaluation set is most likely due to the lack of highly complex and occluded videos in the AIST++ dataset.
The most significant improvement appears on the most complex and occluded choreographies. Upon visual inspection of the performance of the base MediaPipe model on select videos heavy in occlusion and abnormal movements, we observed some detrimental patterns in the base models predictions. The base MediaPipe model’s keypoint predictions consistently collapse and jump randomly.

To visualize the validity of the smoothing pipeline on occluded and complex movements, we create a plot of the pairwise distance error over time for the matched video and an additional mismatched video set. The first figure is the matched videos and the second is the mismatched videos processed by the MediaPipe w/ Kalman + Opt. Flow and Accel. model: Our smoothing model was able to effectively smooth out the keypoint predictions temporally and track the keypoints through tough motion sets while retaining validity by avoiding oversmoothing on mismatched videos. Therefore, we can confidently claim that our model maintains a viable separation in similarity between matched and mismatched frames whilst handling occlusion.

### Real-Time Demo:

A real-time demo between webcam and reference pose using the Keypoint Pairwise Distance Similarity showed promise at capturing and displaying similarity between the pose of the user and a static reference. The keypoint pairwise distance similarity worked decently well for distinguishing scores of similar and non-similar user poses to that of a static reference, though as previously reasoned and shown above, it is not 3D-rotation invariant, and fails completely when exposed to the same pose at a different angle relative to the camera. Because of its sensitivity to small joint angle variations, and rigid, immediate-frame comparison between user and reference, it was difficult to accurately score similarity using a moving video reference, as was otherwise originally intended. The MediaPipe pose model also added significant latency, making the static reference comparison run at around 15 FPS, and the moving video reference at around 8-9 FPS.

## Discussion

### Summary of Results

Our system successfully extended the base MediaPipe model by integrating temporal smoothing with a Kalman filter and optical flow tracking. On the AIST++ dataset, we observed:
- Optical flow contributed to displaying MediaPipe keypoints across occlusion that the model could not represent at high confidence.
- A reduction in pairwise distance error on matched videos (from 5.41 → 2.68).
- Improved embedding similarity on matched poses (from 0.933 → 0.945).
- Although the short fine-tuning run on YOLOv11 produced only a modest accuracy gain, the training dynamics suggest that the model had not yet converged. The loss continued to decrease steadily throughout the 20 epochs, and qualitative inspection showed progressively fewer failures on occluded joints. This indicates that the model was still benefiting from additional exposure to occlusion-augmented samples and had not reached its capacity. With a larger training set, longer training schedule, and more diverse occlusion patterns, YOLOv11 would likely achieve substantially greater robustness.
- The Kalman + Optical Flow variant showed greater stability in complex occlusions compared to the base model. However, the separation between match and mismatch similarity scores remained similar across models, limiting discriminative power in general conditions. Nonetheless, significant gains were achieved under occlusion-heavy sequences, suggesting our refinements are effective where baseline models typically fail.

### Comparison

Unlike prior real-time pose evaluation systems, which primarily rely on frame-by-frame inference and hand-tuned thresholds, our system:

- Uses a motion-aware filtering pipeline for robust temporal smoothing.
- Incorporates a scale- and rotation-invariant similarity metric for fairer cross-subject comparison.
- Handles occlusion and motion jitter better than out-of-the-box MediaPipe.

Other works like DepthCheck focused on task-specific evaluation (e.g. squat depth). Our general-purpose matching framework targets broader pose similarity and reference alignment, offering flexibility but facing complexity in evaluation.

### Future Work
- 3D Keypoint Matching: Extend evaluation to use 3D keypoints for more accurate pose alignment and occlusion robustness.
- Adaptive Similarity Thresholds: Incorporate confidence-aware or motion-aware thresholds to better discriminate between matches and mismatches dynamically.
- Data Augmentation: Create a larger curated dataset with synthetic occlusions and diverse body proportions to strengthen generalization.
- Real-Time Comparison: Delay user pose similarity scoring with current video reference frame to allow user to mimic the reference in real-time without frame-perfect timing, accounting for and potentially dynamically adapting to user’s response time. Find and test lower-latency keypoint models for better real-time processing.

## Conclusion

After a complex analysis of the SOTA models on occlusion, analysis on the benefit of optical flow and the kalman filter on tracking keypoints through occlusion, creating a rotation and scale invariant similarity metric, we successfully achieved a complete end-to-end pose keypoint prediction and similarity estimation pipeline capable of handling the challenges where SOTA models fall short.  With these techniques capable of handling highly complex and variable motion, we believe there is potential transferability of these methods to other alike movements, such as that in sports or other non-traditional movement patterns. 

## References:
1. Ma, S., Zhang, J., Cao, Q., & Tao, D. (2024). PoseBench: Benchmarking the Robustness of Pose Estimation Models under Corruptions. arXiv. https://doi.org/10.48550/arXiv.2406.14367 
2. Shafie, A. A., Kamaru Zaman, F. H., & Ali, M. H. (2009). Motion detection techniques using optical flow. ResearchGate. https://www.researchgate.net/publication/265538405_Motion_Detection_Techniques_Using_Optical_Flow
3. Jiao, Y., Shi, G., & Tran, T. D. (2021). Optical flow estimation via motion feature recovery. arXiv. https://arxiv.org/abs/2101.06333 
4. G. Revach, N. Shlezinger, X. Ni, A. L. Escoriza, R. J. G. van Sloun and Y. C. Eldar, "KalmanNet: Neural Network Aided Kalman Filtering for Partially Known Dynamics," in IEEE Transactions on Signal Processing, https://ieeexplore.ieee.org/document/9733186 
5. Welch, G., & Bishop, G. (2006). An introduction to the Kalman filter. University of North Carolina at Chapel Hill, Department of Computer Science. https://www.cs.unc.edu/~welch/media/pdf/kalman_intro.pdf 
6. Pavllo, D., Feichtenhofer, C., Grangier, D., & Auli, M. (2018). 3D human pose estimation in video with temporal convolutions and semi-supervised training. arXiv.org. https://arxiv.org/abs/1811.11742
7. Li, Y., Zhang, S., Wang, Z., Yang, S., Yang, W., Xia, S., & Zhou, E. (2021, April 8). TokenPose: Learning Keypoint tokens for human pose estimation. arXiv.org. https://arxiv.org/abs/2104.03516