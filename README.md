# MTCNN_API
Convenient API of MTCNN face or face landmark detection.

Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Neural Networks

Tensorflow implementation of the face detection / alignment algorithm found at
https://github.com/kpzhang93/MTCNN_face_detection_alignment

I will going to support more framework such as Caffe, Pytorch and Mxnet.

- [x] Tensorflow
- [] Caffe
- [] Pytorch
- [] Mxnet

# Requirement
* tensorflow
* python2.x or python3.x
* cv2

# API

This is a python package and you should `import MTCNN_API`.

### detect_cv2_ims
MTCNN_API.detect_cv2_ims(images,minsize=20,threshold=(0.6, 0.7, 0.7),scale_factor=0.709,gpu_fraction=0.5)

* images: a list contains cv2 opened images
* minsize: minimum size of face
* threshold: three steps's threshold
* scale_factor: scale factor
* gpu_fraction: tensorflow gpu_fraction

return: (boxes, landmarks)
* boxes is a list contains ndarray with shape (n_faces_in_pic, 5), 5 number represent for (x1,y1,x2,y2,score).
* landmarks is a list contains ndarray with shape(n_faces_in_pic, 10),10 landmark number represent for
        (leyex,reyex,nosex,lmouthx,rmouthx,leyey,reyey,nosey,lmouthy,rmouthy) l:Left, r:Right

# Example

    import cv2
    import MTCNN_API

    images = []
    for file_name in ['test1.jpg', 'test2.png', 'test3.png']:
        im = cv2.imread(file_name)
        images.append(im)

    boxes, landmarks = MTCNN_API.detect_cv2_ims(images)