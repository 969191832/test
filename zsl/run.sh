tmp=osnet_x1_0_mod_two_branch.yaml
# tmp=osnet_x1_0_mod_single.yaml
# tmp=osnet_x1_0_mod_parts.yaml
# tmp=im_r50_softmax_256x128_amsgrad.yaml
# tmp=test.yaml
# tmp=eval.yaml
# tmp=resnet_pcb.yaml
python ../scripts/main.py \
	--config-file ../configs/$tmp \
	--transforms random_flip random_erase \
	--root /data2/zsl/dataset
