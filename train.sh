set -ex
python train.py --dataroot ./datasets_fine/font --model lsg_fcst --name result_LSG --save_epoch_freq 1 --gpu_ids 0 --batch_size 16 --display_port 8099 --netG LSG_MLAN --dataset_mode font  --sanet multi
