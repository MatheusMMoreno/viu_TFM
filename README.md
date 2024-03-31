This is a repository to store information related to VIU Artificial Intelligence TFM/2024


## Training & Validation Dataset:

Training Dataset and models used in this report was extracted from [Github Repository](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_)  &  [Github Repository](https://github.com/timesler/facenet-pytorch/tree/master)

The Dataset used was [Casia-Webface](https://drive.google.com/file/d/1KxNCrXzln0lal3N4JiYl9cFOIhT78y1l/view) [1]

If there are any questions about how to convert the database to Folder/class/image.jpg format you can check the code on Extract_to_images about how to do it. It is necessary to install specific packages to be able to export the images correctly. Code check_images was used to validate the output of the images.

consult Pytorch availability for CUDA and make sure to download the proper packages to ensure better performance.   Details are in this [Pytorch](https://pytorch.org/get-started/locally/) page. The code on check_cuda can check if you have enabled the GPU.

Trained models are in this one drive folder [Trained Models](https://1drv.ms/f/s!Ap41Q28xA5OT0W6iXNrVr_VKYKP0?e=g2gzmL)


## Reference

[1] Dong Yi, Zhen Lei, Shengcai Liao, Stan Z. Li. Learning Face Representation from Scratch. arXiv:1411.7923, 2014.

[2] M. Sandler, A. Howard, M. Zhu, A. Zhmoginov, and L.-C. Chen. Mobilenetv2: Inverted residuals and linear bottlenecks. In Proceedings of the IEEE Conference on Computer, 2014.

[3] S. Yang, P. Luo, C. C. Loy, and X. Tang, "MobileFaceNets: Efficient CNNs for Accurate Real-Time Face Verification on Mobile Devices," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

[4] Deng, Z.-Y., Chiang, H.-H., Kang, L.-W., Li, H.-C.: A lightweight deep learning model for real-time face recognition. IET Image Process. 17, 3869–3883 (2023). https://doi.org/10.1049/ipr2.12903
