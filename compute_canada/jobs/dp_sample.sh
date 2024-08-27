cd ../..

SAMPLE_FLAGS="
--timestep_respacing 250"

MODEL_FLAGS="
--attention_resolutions 32,16,8 
--class_cond False
--diffusion_steps 1000
--image_size 256
--learn_sigma True
--noise_schedule linear
--num_channels 256
--num_head_channels 64
--num_res_blocks 2
--resblock_updown True
--use_fp16 True
--use_scale_shift_norm True"

python scripts/DP_sample.py \
    $MODEL_FLAGS \
    --model_path models/256x256_diffusion_uncond.pt \
    --image_path ref_images \
    --guide_scales "2.5, 3.0, 3.5, 4.0, 4.5, 5.0" \
    $SAMPLE_FLAGS
