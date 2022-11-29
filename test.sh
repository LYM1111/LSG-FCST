set -ex
python test.py --dataroot ./datasets_fine/font  --model lsg_fcst  --gpu_ids 0  --name result_ftransgan  --phase unseen_font_unseen_character --dataset_mode font
python test.py --dataroot ./datasets_fine/font  --model lsg_fcst  --gpu_ids 0  --name result_ftransgan  --phase unseen_font_seen_character --dataset_mode font
python test.py --dataroot ./datasets_fine/font  --model lsg_fcst  --gpu_ids 0  --name result_ftransgan  --phase seen_font_unseen_character --dataset_mode font
