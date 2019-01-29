# Deep Neural Network for Speech Enhancement:

Most of the Automatic Speech Recognition (ASR) systems such as Siri, Alexa and Google Home are trained on clean recording and introducing noise can significantly reduce the performance of these systems because system hasn't seen this specific noise during training. A workaround solution would be to enhance the quality of signal before feeding it to ASR systems. In this project, we explore and compare the outputs various Deep Neural Network architectures. The code has been implemented using Pytorch. The preprocessing part is skipped here, we directly use the Matlab file provided as input. I haven't pushed the training set here, because it is a really huge file.

The training set consists of 2000 utterances and each utterance is a sequence of variable length. The input features have 246 dimensions and target labels have 64 dimensions.

The testing set consists of 192 utterances and each utterance is a sequence of variable length. The input features have 246 dimensions and target labels have 64 dimensions.


The project is based on the paper below:

http://web.cse.ohio-state.edu/~wang.77/papers/WNW.taslp14.pdf

The final predicted output is converted into a numpy array and then calculate Short-Time Objective Intelligibility (STOI) Score, Perceptual Evaluation of Speech Quality (PESQ) Score to measure quality and understandability of the noisy audio. 


