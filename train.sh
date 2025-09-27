python train.py  \
--dataset Spines \
--list_dir '/home/zdt/CS-Unet-main/lists/lists_Spines' \
--root_path '/home/zdt/MCF-main/dataset/Spine_modified' \
--output_dir './results/transunetqing' \
--max_epochs 300 \
--img_size 224 \
--base_lr 0.05 \
--num_classes 16 \
--batch_size 24 \


#python train.py  \
#--dataset Spines \
#--list_dir '/home/zdt/CS-Unet-main/lists/lists_Spines' \
#--root_path '/home/zdt/MCF-main/dataset/Spine_modified' \
#--output_dir './results/para+0.1+MGRM' \
#--max_epochs 300 \
#--img_size 224 \
#--base_lr 0.05 \
#--num_classes 16 \
#--batch_size 24 \


#--dataset Synapse \
#--cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
#--max_epochs 300 \
#--img_size 224 \
#--base_lr 1E-3 \
#--base_weight 5E-4 \
#--batch_size 24










#--volume_path '/home/zdt/MCF-main/dataset/Spine/test' \
# To add acdc, revise train.py, trainer.py, dataset_synapse.py, my_train.sh. and add metrics.py