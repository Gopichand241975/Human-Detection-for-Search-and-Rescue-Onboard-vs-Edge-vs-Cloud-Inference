# Human-Detection-for-Search-and-Rescue-Onboard-vs-Edge-vs-Cloud-Inference
This project focuses on human detection for Search and Rescue (SAR) operations using drones and the YOLOv8n deep learning model. The model was trained on the VisDrone dataset and deployed across three different inference architectures: onboard inference using a Raspberry Pi 3 Model B+, edge inference using a laptop, and cloud inference. A total of 300 test images were used for evaluation, and five experimental runs were conducted for each deployment scenario to ensure consistent results. The system was evaluated based on inference latency and power consumption, while the output consisted of human detections with bounding boxes drawn around detected individuals in aerial images. The comparative analysis helps identify the most efficient architecture for real-time SAR drone applications.

##Observations
- Onboard inference has minimal communication delay but limited computational power.
- Cloud inference has high computational capability but suffers from network latency.
- Edge inference achieves the best trade-off between computation and communication, resulting in the most efficient performance in this study.
  
## Inference Output

![Inference Output](Output/inference_output.jpg)

## Latency Comparison

![Latency Comparison of Three Scenarios](Output/latency%20comparision%20of%20three%20scenarios.jpg)

## Average Latency Breakdown

![Average Latency Breakdown by Scenario](Output/average%20latency%20breakdown%20by%20scenario.jpg)
