# Optimized-VNECT-pose-estimation-network
The source code is related to the master thesis topic titled "Optimization of VNECT network for 2D upper body pose estimation". The main aim of this work is to optimize a pre-trained pose estimation network for faster inference. The thesis is inspired by the VNECT network, which was the first method to capture stable and temporally consistent 3D skeletal pose of a human from videos. The original implementation was in caffe, please contact the authors of the paper for model weights (http://gvv.mpi-inf.mpg.de/projects/VNect/).

To use the sourcecode, we need the following packages in our Ubuntu 16.04 environment
•	TensorFlow 1.14.0
•	pycaffe
•	python 3.x
•	OpenCV 3.x
•	TensorRT

The files are organized as follows:
•	Drop the caffe weights (.prototxt and .caffemodel) obtained from the original authors in the folder “./model/caffe_model”
•	Run “initialize_weights.py” to convert the weights in caffe to pickle and tensorflow checkpoint files.
•	To test the baseline performance of the model, run “baseline_vnect.py”
•	To change the pre-trained network to optimized model run “optimization.py”.
•	Use the optimized models directly to run the inference for pose estimation on images and videos by executing “final_vnect_images.py” and “final_vnect_videos.py” respectively.

Reference Repositories:
timctho/Vnect-tensorflow
EJShim/vnect_estimator
