LAUNCH="accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29531"

algorithm=${2}
f_type=${3}
prompt_template="BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:"
dataset_name="PKU-Alignment/PKU-SafeRLHF-10K"
preference="better"

if [ "${f_type}" != "reverse_kl" ]; then
    type_str="-${f_type}"
else
    type_str=""
fi

max_length=512
num_beams=1
chosen_only=False
sanity_check=False
output_dir="/ext_hdd/sjkim/mod/dpo"
sft_model_name="PKU-Alignment/alpaca-7b-reproduced"
dpo_model_1_name="${output_dir}/dpo-better/best_checkpoint"
dpo_model_2_name="${output_dir}/dpo-safer/best_checkpoint"
seed=42

dpo_run_name="${dataset_name}/dpo${type_str}"

# weight_1: 0.0 -> 1.0
# weight_2: 1.0 -> 0.0
for weights in \
    "0.1 0.9" \
    "0.2 0.8" \
    "0.4 0.6" \
    "0.6 0.4" \
    "0.8 0.2" \
    "0.9 0.1"
do
    weight_1=$(echo ${weights} | awk '{print $1}')
    weight_2=$(echo ${weights} | awk '{print $2}')

    echo "=========================================="
    echo "[RUN] weight_1=${weight_1}, weight_2=${weight_2}"
    echo "=========================================="

    PYTHONPATH=. CUDA_VISIBLE_DEVICES=${1} $LAUNCH scripts/eval/${algorithm}.py \
        --f_type ${f_type} \
        --prompt_template "${prompt_template}" \
        --sft_model_name ${sft_model_name} \
        --dpo_model_1_name ${dpo_model_1_name} \
        --dpo_model_2_name ${dpo_model_2_name} \
        --weight_1 ${weight_1} \
        --weight_2 ${weight_2} \
        --num_beams ${num_beams} \
        --dataset_name "${dataset_name}-${preference}" \
        --sanity_check ${sanity_check} \
        --max_length ${max_length} \
        --training_args.output_dir "${output_dir}/${dpo_run_name}" \
        --training_args.run_name ${dpo_run_name} \
        --peft_config.target_modules q_proj k_proj v_proj o_proj \
        --seed ${seed}
done