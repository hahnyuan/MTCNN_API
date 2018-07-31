from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import os
from . import detect_face

_MTCNN=[]

def get_net(gpu_fraction=0.5):
    global _MTCNN
    if len(_MTCNN)!=3:
        with tf.Graph().as_default():
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
            sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
            with sess.as_default():
                pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(os.path.split(__file__)[0],'weights'))

            _MTCNN+=[pnet,rnet,onet]
    else:
        pnet, rnet, onet=_MTCNN
    return pnet,rnet,onet


def detect_cv2_ims(images,minsize=20,threshold=(0.6, 0.7, 0.7),scale_factor=0.709,gpu_fraction=0.5):
    """
    :param images: a list contains cv2 opened images
    :param minsize: minimum size of face
    :param threshold: three steps's threshold
    :param scale_factor: scale factor
    :param gpu_fraction: tensorflow gpu_fraction
    :return: (boxes, landmarks),
    boxes is a list contains ndarray with shape (n_faces_in_pic, 5), 5 number represent for (x1,y1,x2,y2,score).
    landmarks is a list contains ndarray with shape(n_faces_in_pic, 10),10 landmark number represent for
        (leyex,reyex,nosex,lmouthx,rmouthx,leyey,reyey,nosey,lmouthy,rmouthy) l:Left, r:Right
    """
    pnet, rnet, onet=get_net(gpu_fraction)
    if type(images) is not list:
        images=[images]
    boxes=[]
    landmarks=[]
    for bgr in images:
        try:
            rgb = bgr[..., ::-1]
            box, landmark = detect_face.detect_face(rgb, minsize, pnet, rnet, onet, threshold, scale_factor)
        except:
            box=[]
            landmark=[]
        boxes.append(box)
        landmarks.append(landmark)
    return boxes,landmarks