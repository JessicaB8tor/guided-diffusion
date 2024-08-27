cd ../..
python scripts/classifier_sample.py \
  --attention_resolutions 32,16,8 \
  --class_cond False \
  --diffusion_steps 1000 \
  --dropout 0.1 \
  --use_ddim False \
  --image_size 256 \
  --batch_size 4 \
  --learn_sigma True \
  --noise_schedule linear \
  --num_channels 256 \
  --num_head_channels 64 \
  --num_res_blocks 2 \
  --resblock_updown True \
  --use_new_attention_order True \
  --use_fp16 True \
  --use_scale_shift_norm True \
  --classifier_scales "1e-6, 5e-6, 1e-5" \
  --classifier_path models/256x256_classifier.pt  \
  --classifier_depth 4 \
  --model_path models/256x256_diffusion_uncond.pt