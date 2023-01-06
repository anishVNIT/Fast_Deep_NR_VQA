# Fast_Deep_NR_VQA
An End-to-End Fast No-Reference Video Quality Predictor with Spatiotemporal Feature Fusion

This work proposes a reliable and efficient end-to-end No-
Reference Video Quality Assessment (NR-VQA) model that fuses deep
spatial and temporal features. Since both spatial (semantic) and temporal
(motion) features have a significant impact on video quality, we have
developed an effective and fast predictor of video quality by combining
both. ResNet-50, a well-known pre-trained image classification model, is
employed to extract semantic features from video frames, whereas I3D, a
well-known pre-trained action recognition model, is used to compute spatiotemporal
features from short video clips. Further, extracted features
are passed through a regressor head that consists of a Gated Recurrent
Unit (GRU) followed by a Fully Connected (FC) layer.

For feature extraction use i3d_resnet_feature_extraction_main.py

For training and testing the model use Score_Prediction_I3D_ResNet.py
