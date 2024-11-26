ckpt=$1
output_dir=$2

mkdir -p $output_dir

fairseq-generate data/ctcs2ut/fr-en/unit_raw \
    --user-dir CTCS2UT \
    --config-yaml config.yaml --gen-subset train --task speech_to_text_modified \
    --path $ckpt \
    --beam 10 --max-len-a 1 --max-tokens 20000 --scoring sacrebleu \
    --required-batch-size-multiple 1 \
    --results-path $output_dir

grep "^D\-" $output_dir/generate-train.txt | \
  sed 's/^D-//ig' | sort -nk1 | cut -f3 \
  > $output_dir/generate-train.unit

python CTCS2UT/scripts/replace_unit.py \
    --input-tsv data/ctcs2ut/fr-en/unit_raw/train.tsv \
    --unit-txt $output_dir/generate-train.unit \
    --output-tsv $output_dir/train.tsv