exp=$1

ROOT=~/speech2speech
VOCODER_CKPT=vocoder/mhubert_lyr11_km1000_en/g_00500000
VOCODER_CFG=vocoder/mhubert_lyr11_km1000_en/config.json

checkpoint_dir=checkpoints/$exp
output_dir=results/$exp

mkdir -p $output_dir

python fairseq/scripts/average_checkpoints.py \
    --inputs $checkpoint_dir/checkpoint.best_loss*.pt \
    --output $checkpoint_dir/average_best_checkpoint.pt

fairseq-generate data/ctcs2ut/fr-en/unit_kd \
    --user-dir CTCS2UT \
    --config-yaml config.yaml --gen-subset test --task nat_speech_to_text_ctc \
    --src-upsample-ratio 2 \
    --path $checkpoint_dir/average_best_checkpoint.pt \
    --iter-decode-max-iter 0 --iter-decode-eos-penalty 0 \
    --required-batch-size-multiple 1 \
    --max-tokens 40000 --scoring sacrebleu > $output_dir/output.txt

grep "^D\-" $output_dir/output.txt | \
  sed 's/^D-//ig' | sort -nk1 | cut -f3 \
  > $output_dir/output.unit

python CTCS2UT/scripts/postprocess.py \
    --input-file $output_dir/output.unit \
    --output-file $output_dir/output.unit.reduce

python fairseq/examples/speech_to_speech/generate_waveform_from_code.py \
  --in-code-file $output_dir/output.unit.reduce \
  --vocoder $VOCODER_CKPT --vocoder-cfg $VOCODER_CFG \
  --results-path $output_dir --dur-prediction

cd asr_bleu/
python compute_asr_bleu.py \
  --lang en \
  --audio_dirpath $ROOT/$output_dir \
  --reference_path $ROOT/data/ctcs2ut/fr-en/unit_raw/test.txt \
  --reference_format txt