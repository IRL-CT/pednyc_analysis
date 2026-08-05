# Optional skeleton encoders

## ST-GCN

ST-GCN expects a stable graph-ordered tensor (N,C,T,V,M) and a declared joint adjacency graph. The 20-joint Unity humanoid mapping and adjacency used here are technically convertible, but finger/controller availability varies and there are too few positive trials for defensible training or fine-tuning. No ST-GCN was forced.

## MotionBERT

MotionBERT generally expects a Human3.6M-like 17-joint 2D/3D representation and pretrained motion semantics. Mapping is possible only approximately (Unity LowerArm/LowerLeg stand in for elbow/knee segments), camera/projective inputs are absent, and its pretrained objectives do not match a sparse hand-gesture trial label. Fine-tuning is not defensible for this sample without an external validation cohort.

## SMPL / SMPL-X

SMPL/SMPL-X or Mocap-to-SMPLX could standardize skeleton geometry, but they are not behavior classifiers. Their extra dependencies, licenses, fitting uncertainty, and potential loss of controller-hand detail outweigh their value for this first experiment.
