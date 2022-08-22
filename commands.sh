DATADIR=/home/amar/Desktop/workspace/barcelona

conda develop .



python utils/thirdpartyscripts/infer_subimages.py --config-file COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml --output $DATADIR/detectron --image-ext jpg --input $DATADIR/images/



python3 demo/calibrate_video.py --path_to_data $DATADIR

OPENPOSEDIR=/home/amar/Desktop/workspace/openpose-1.1.0
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64/
python3 demo/estimate_poses.py --path_to_data $DATADIR --openpose_dir $OPENPOSEDIR