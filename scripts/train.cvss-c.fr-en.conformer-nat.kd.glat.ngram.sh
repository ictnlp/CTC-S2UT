exp=cvss-c.fr-en.conformer-nat.kd.glat.ngram
mkdir -p checkpoints/$exp/
cp checkpoints/cvss-c.fr-en.conformer-nat.kd.glat/average_best_checkpoint.pt checkpoints/$exp/checkpoint_last.pt
fairseq-train data/ctcs2ut/fr-en/unit_kd \
  --user-dir CTCS2UT \
  --config-yaml config.yaml \
  --task nat_speech_to_text_ctc --noise full_mask \
  --arch s2t_conformer_ctc_ngram_glat --share-decoder-input-output-embed \
  --pos-enc-type rel_pos --decoder-learned-pos --attn-type espnet \
  --apply-bert-init \
  --encoder-layers 12 --encoder-embed-dim 256 --encoder-ffn-embed-dim 2048 --encoder-attention-heads 4 \
  --decoder-layers 6 --decoder-embed-dim 512 --decoder-ffn-embed-dim 2048 --decoder-attention-heads 8 \
  --max-source-positions 6000 --max-target-positions 2048 --src-upsample-ratio 2 --plain-ctc \
  --criterion nat_loss_ngram_glat --use-ngram --ngram-size 2 \
  --src-embedding-copy \
  --glat-p 0.3:0.3@6k \
  --optimizer adam --adam-betas '(0.9,0.98)' --fp16 \
  --label-smoothing 0.01 --weight-decay 0.01 --dropout 0.3 --attention-dropout 0.3 --relu-dropout 0.3 \
  --lr-scheduler inverse_sqrt  --warmup-updates 500 \
  --clip-norm 1.0 --lr 3e-4 --warmup-init-lr 1e-7 --stop-min-lr 1e-9 \
  --ddp-backend=legacy_ddp --patience 10 \
  --max-tokens 2000 --update-freq 160 --grouped-shuffling \
  --max-update 6000 --max-tokens-valid 2000 \
  --save-interval 1 --save-interval-updates 200 \
  --seed 1 \
  --train-subset train --valid-subset dev \
  --validate-interval 1000 --validate-interval-updates 200 \
  --save-dir checkpoints/$exp \
  --keep-best-checkpoints 5 \
  --keep-interval-updates 5 --keep-last-epochs 5 \
  --no-progress-bar --log-format json --log-interval 10 \
  --num-workers 0