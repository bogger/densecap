qlua evaluate_from_file.lua \
-data_h5 ./data/VG-regions.h5 \
-data_json ./data/VG-regions-dicts.json \
-res_json ../caffe_s2vt/output/faster_rcnn_end2end/vg_test/faster_rcnn_cap_two_stage_iter_150000/results.json \
-gpu 0 \
-split test \
-max_images -1 \
-vis 1 
