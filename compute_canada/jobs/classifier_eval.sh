cd ../..
python scripts/eval_classifier_accuracy.py \
    --dataset $SLURM_TMPDIR/imagenet \
    --use_fp16 True