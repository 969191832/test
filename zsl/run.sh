yaml_file=im_osnet_x1_0_softmax_256x128_amsgrad.yaml

python ../scripts/main.py \
	--config-file configs/$yaml_file \
	--transforms random_flip random_erase \
	--root /data2/zsl/dataset
