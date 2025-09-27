python test.py  \
--num_classes 16 \
--volume_path '/home/zdt/MCF-main/dataset/Spine_modified' \
--dataset Spines \
--list_dir './lists/lists_Spines' \
--output_dir './results/transunetqing' \
--test_save_dir './results/transunetqing' \
--max_epochs 150 \
--img_size 224 \
--is_savenii


#python tests.py  \
#--dataset_name Spines \
#--cfg configs/cswin_tiny_224_lite.yaml \
#--list_dir './lists/lists_Spines' \
#--output_dir './results/spines16' \
#--test_save_dir './results/spines16/predictions' \
#--max_epochs 150 \
#--num_classes 16 \
#--img_size 224 \
#--is_savenii


#--dataset Synapse \
#--cfg configs/swin_tiny_patch4_window7_224_lite.yaml \
#--max_epochs 300 \
#--img_size 224 \
#--is_savenii
#--volume_path '/home/zdt/MCF-main/dataset/Spine'\
#--volume_path '/home/zdt/MCF-main/dataset/Spine'\







# To add acdc, revise train.py, trainer.py, dataset_synapse.py, my_train.sh. and add metrics.py