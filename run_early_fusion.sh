#! /bin/zsh

EPOCH=50
dataset='v'

### 1表示true, 0表示false
pre_trained_descs_norm_flag=0
dimension=4096
#####norm, nonorm
norm_flag='nonorm'

####PCA, RP, AUTO, no_reduce
reduce_flag=1
reduce_method='AUTO'

####alexnet, resnet101, densenet169
pre_trained_model_name='densenet169'

####cat, CCA, AE, no_fusion
fusion_flag=1
fusion_method='AE'
####sigmoid, cnn, tanh, no
fusion_AE_flag='sigmoid'
####0, 128, 256, 其中0表示没有融合
fusion_dimension=256
####AE fusion train epoch
fusion_epoch=500

L2_norm="noL2norm"
batch_size=32
lr=0.0001

####-u是为了禁止缓存，让结果可以直接进入日志文件
while [ ${fusion_dimension} != '512' ]
do
######auto encoder 预训练模型地址
####alexnet, resnet, densenet
####all_descs for 50, original for 100
autoencoder_parameter_path="../../utils/parameters/model_parameters/${pre_trained_model_name}/${EPOCH}_epoch/all_descs/$dataset/${EPOCH}_${dataset}_${pre_trained_model_name}_${dimension}_autoencoder_cnn.pth"

echo "Auto-Encoder降维模型参数地址:${autoencoder_parameter_path}"

#####训练数据集地址(PCA, Random Project)
pre_trained_descs_path="../../utils/parameters/descriptors/${pre_trained_model_name}/${dataset}_${model_name}_all_descs.npy"

(CUDA_VISIBLE_DEVICES=1  python -u  run_early_fusion.py \
--dataset ${dataset} \
--pre_trained_descs_norm_flag ${pre_trained_descs_norm_flag} \
--dimension ${dimension} \
--reduce_flag ${reduce_flag} \
--reduce_method ${reduce_method} \
--autoencoder_parameter_path ${autoencoder_parameter_path} \
--pre_trained_model_name ${pre_trained_model_name} \
--pre_trained_descs_path ${pre_trained_descs_path} \
--fusion_flag ${fusion_flag} \
--fusion_method ${fusion_method} \
--fusion_AE_flag ${fusion_AE_flag} \
--fusion_dimension ${fusion_dimension} \
--fusion_epoch ${fusion_epoch} \
--batch_size ${batch_size} \
--lr ${lr} \
> "./log/${pre_trained_model_name}/${EPOCH}_train_test_${dataset}_${dimension}_${pre_trained_model_name}_${norm_flag}_${reduce_method}_${fusion_method}_${fusion_AE_flag}_${L2_norm}_${fusion_dimension}_${fusion_epoch}_${batch_size}_${lr}.log"
)

#dimension=$((dimension*2))
fusion_dimension=$((fusion_dimension*2))
done
