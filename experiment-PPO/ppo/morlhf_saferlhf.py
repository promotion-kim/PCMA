import os
import time
from dataclasses import dataclass, field
from typing import Optional
from accelerate import Accelerator
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, set_seed
import numpy as np
import pandas as pd
from utils import print_trainable_parameters, load_main_tokenizer, Instructions, Instructions_summary, \
                  build_dataset, build_dataset_summary             
from multi_reward_models import RewardModels
from safe_rlhf.models import AutoModelForScore
tqdm.pandas()
from peft import LoraConfig
import matplotlib.pyplot as plt
import wandb

from src.data.configs import DATASET_CONFIGS, DEFAULT_PROMPT_TEMPLATE
from datasets import Dataset

# define paths for two datasets
hhrlhf_dataset_path = 'Anthropic/hh-rlhf'
summary_dataset_path = 'openai/summarize_from_feedback'

@dataclass
class ScriptArguments:
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    disable_wandb: Optional[str] = field(default=False, metadata={'help': 'Whether to disable wandb or not.'})
    save_directory: Optional[str] = field(default='./logs_morlhf/')
    epochs: Optional[int] = field(default=1, metadata={'help': "Number of training epoches"})
    learning_rate: Optional[float] = field(default=1e-5, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=1, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=4, metadata={"help": "the batch size64"})
    gradient_accumulation_steps: Optional[int] = field(default=1, metadata={"help": "the number of gradient accumulation steps"})
    early_stopping: Optional[bool] = field(default=True, metadata={"help": "whether to early stop"})
    target: Optional[float] = field(default=3, metadata={"help": "target kl divergence of adaptive control"})
    init_kl_coef: Optional[float] = field(default=0.2,metadata={"help": "0.05 Initial KL penalty coefficient (used for adaptive and linear control)"},)
    max_grad_norm: Optional[float] = field(default=0.5, metadata={"help": "Maximum gradient norm for gradient clipping"})
    load_in_8bit: Optional[bool] = field(default=True, metadata={"help": "loading model in 8 bit or bfloat16"})
    preference: Optional[float] = field(default=0.5, metadata={"help": "the weight for reward 1"})
    wandb_name: Optional[str] = field(default='morlhf_llamma2_klreg0.2', metadata={"help": "Name for this experiment"})
    base_model_name: Optional[str] = field(default='./merged_sft_summary', metadata={'help':"the path to the sft model; need to merge if using lora"})
    reward_names:Optional[str] = field(default='harmless,helpful,humor') 
    exp_type: Optional[str] = field(default='assistant', metadata={"help": "exp type, 'summary' or 'assistant' "})
    dataset_name: Optional[str] = field(
        default="PKU-Alignment/PKU-SafeRLHF-10K-better",
        metadata={"help": "dataset name, aligned with MODPO"}
    )
    prompt_template: Optional[str] = field(
        default="BEGINNING OF CONVERSATION: USER: {raw_prompt} ASSISTANT:",
        metadata={"help": "prompt template"}
    )
    max_prompt_length: Optional[int] = field(default=512)
    max_new_tokens: Optional[int] = field(default=128)
    reward_model_max_length: Optional[int] = field(default=512)
    seed: Optional[int] = field(default=8888, metadata={"help": "random seed"})
    train_split: Optional[str] = field(default="train", metadata={"help": "split used for PPO training prompts"})
    eval_split: Optional[str] = field(default="validation", metadata={"help": "split used for periodic validation generation"})
    eval_steps: Optional[int] = field(default=100, metadata={"help": "run validation generation every N PPO update steps; set <=0 to disable"})
    eval_num_prompts: Optional[int] = field(default=5, metadata={"help": "number of validation prompts to generate for wandb table"})
    eval_max_new_tokens: Optional[int] = field(default=128, metadata={"help": "max new tokens for periodic validation generation"})

parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
exp_type = script_args.exp_type
preference = [round(script_args.preference, 1), round(1 - script_args.preference, 1)]
script_args.wandb_name = script_args.wandb_name + '_pref{}_{}'.format(preference[0], preference[1])

tokenier_name = script_args.base_model_name
base_model_name = script_args.base_model_name
print('base model: ', base_model_name)

if script_args.disable_wandb: # if you don't need the wandb log
    os.environ['WANDB_DISABLED'] = 'true' 

reward_names = [x.strip() for x in script_args.reward_names.split(',')]
num_rewards = len(reward_names)
print('number of rewards: {}'.format(num_rewards))
#######################################################################
if num_rewards == 3:
    preference = [round(1 / num_rewards, 2) for _ in range(num_rewards)]
print('preference: {}'.format(preference))
#######################################################################
reward_path_tokenizer_dict = {
    # helpfulness reward: higher is better
    "helpful": ["PKU-Alignment/beaver-7b-v1.0-reward"],

    # safety cost: higher means more unsafe, so we will multiply by -1 later
    "harmless": ["PKU-Alignment/beaver-7b-v1.0-cost"],
}

reward_sign_dict = {
    "helpful": 1.0,
    "harmless": -1.0,
}

reward_model_path_list = []
rm_tokenizer_path_list = []
reward_sign_list = []

for name in reward_names:
    if name not in reward_path_tokenizer_dict.keys():
        raise NotImplementedError(f"Unknown reward name: {name}")

    reward_model_path_list.append(reward_path_tokenizer_dict[name][0])
    rm_tokenizer_path_list.append(reward_path_tokenizer_dict[name][0])
    reward_sign_list.append(reward_sign_dict.get(name, 1.0))

print("reward model paths:", reward_model_path_list)
print("reward signs:", reward_sign_list)

os.makedirs(os.path.join(script_args.save_directory, script_args.wandb_name), exist_ok=True)


config = PPOConfig(
    model_name=base_model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    early_stopping=script_args.early_stopping,
    target=script_args.target,
    max_grad_norm=script_args.max_grad_norm,
    optimize_cuda_cache=True,
    init_kl_coef=script_args.init_kl_coef,
    tracker_project_name='pcma',
    tracker_kwargs={"wandb":{"name":script_args.wandb_name}},
)

accelerator = Accelerator()
process_id = Accelerator().local_process_index 
gpu_id = process_id
print('process: {}, model gpu id: {}'.format(process_id, gpu_id))

class BeaverScoreModels:
    def __init__(self, model_paths, tokenizer_paths, gpu_id, max_length=1024):
        self.model_paths = model_paths
        self.max_length = max_length
        self.device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")

        self.models = []
        self.tokenizers = []

        print("Loading Beaver score models with AutoModelForScore...")

        for model_path, tokenizer_path in zip(model_paths, tokenizer_paths):
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path,
                use_fast=True,
                trust_remote_code=True,
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "right"

            model = AutoModelForScore.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                device_map={"": gpu_id},
            )
            model.eval()

            self.tokenizers.append(tokenizer)
            self.models.append(model)

        print("Loaded Beaver score models:", model_paths)

    @torch.no_grad()
    def get_reward_model_scores(self, queries_responses, *args, **kwargs):
        texts = []
        for query, response in queries_responses:
            texts.append(str(query) + str(response))

        all_scores = []

        for model, tokenizer in zip(self.models, self.tokenizers):
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=script_args.reward_model_max_length,
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = model(**inputs)

            scores = getattr(outputs, "end_scores", None)
            if scores is None:
                scores = getattr(outputs, "scores", None)
            if scores is None:
                raise RuntimeError(
                    f"AutoModelForScore output has no end_scores/scores. "
                    f"Available fields: {outputs}"
                )

            if scores.ndim == 2 and scores.shape[-1] == 1:
                scores = scores.squeeze(-1)
            elif scores.ndim >= 2:
                scores = scores[:, -1]

            all_scores.append(scores.float().detach().cpu().tolist())

        return all_scores
    
############## load reward models
if any("beaver-7b" in path for path in reward_model_path_list):
    reward_model = BeaverScoreModels(
        reward_model_path_list,
        rm_tokenizer_path_list,
        gpu_id,
        max_length=script_args.reward_model_max_length,
    )
else:
    reward_model = RewardModels(reward_model_path_list, rm_tokenizer_path_list, gpu_id)

rm_tokenizer = AutoTokenizer.from_pretrained(
    rm_tokenizer_path_list[0],
    use_fast=True,
    trust_remote_code=True,
)
if rm_tokenizer.pad_token is None:
    rm_tokenizer.pad_token = rm_tokenizer.eos_token

def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])

set_seed(script_args.seed)
current_device = Accelerator().local_process_index
print(current_device)

lora_config = LoraConfig(
    r=64, 
    lora_alpha=128, 
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


tokenizer = AutoTokenizer.from_pretrained(
    tokenier_name,
    use_fast=True,
    trust_remote_code=True,
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

tokenizer.padding_side = "left"

def build_pku_prompt_dataset(script_args, tokenizer, split="train"):
    rdp = DATASET_CONFIGS[script_args.dataset_name](
        prompt_template=script_args.prompt_template,
        sanity_check=script_args.sanity_check if hasattr(script_args, "sanity_check") else False,
    )

    pref_dataset = rdp.get_preference_dataset(split=split)

    rows = []
    seen = set()

    for ex in pref_dataset:
        # DATASET_CONFIGS 구현에 따라 key 이름이 다를 수 있어서 방어적으로 처리
        raw_prompt = (
            ex.get("raw_prompt")
            or ex.get("prompt")
            or ex.get("query")
            or ex.get("input")
        )

        if raw_prompt is None:
            # 이미 prompt_template이 적용된 필드가 있을 가능성
            raw_prompt = ex.get("text")

        if raw_prompt is None:
            raise KeyError(f"Cannot find prompt key. Available keys: {list(ex.keys())}")

        raw_prompt = str(raw_prompt)

        # 이미 template이 적용되어 있으면 그대로 사용
        if "{raw_prompt}" in script_args.prompt_template:
            query = script_args.prompt_template.format(raw_prompt=raw_prompt)
        else:
            query = raw_prompt

        if query in seen:
            continue
        seen.add(query)

        toks = tokenizer(
            query,
            truncation=True,
            max_length=script_args.max_prompt_length,
            padding=False,
        )

        rows.append({
            "query": query,
            "input_ids": toks["input_ids"],
        })

    return Dataset.from_list(rows)

train_dataset = build_pku_prompt_dataset(script_args, tokenizer, split=script_args.train_split)
eval_dataset = build_pku_prompt_dataset(script_args, tokenizer, split=script_args.eval_split)

# Use deterministic shuffling for PPO prompts and validation prompt samples.
train_dataset = train_dataset.shuffle(seed=script_args.seed)
if len(eval_dataset) > 0:
    eval_dataset = eval_dataset.shuffle(seed=script_args.seed)

class PKUInstructions:
    def get_input(self, text):
        if "ASSISTANT:" in text:
            return text.split("ASSISTANT:")[0].strip() + "ASSISTANT:"
        return text

    def get_response(self, text):
        if "ASSISTANT:" in text:
            return text.split("ASSISTANT:", 1)[1].strip()
        return text

instructions = PKUInstructions()

print(f"Size of the train set ({script_args.train_split}): {len(train_dataset)}")
print(f"Size of the eval set ({script_args.eval_split}): {len(eval_dataset)}")


if script_args.load_in_8bit:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model_name,
        load_in_8bit=True,
        peft_config=lora_config,
        device_map=gpu_id,
    )
else:
    model = AutoModelForCausalLMWithValueHead.from_pretrained(
        base_model_name,
        torch_dtype=torch.bfloat16,
        peft_config=lora_config,
        device_map=gpu_id,
    )

print_trainable_parameters(model)
model.pretrained_model.resize_token_embeddings(len(tokenizer))
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

ppo_trainer = PPOTrainer(
    config, model, tokenizer=tokenizer, dataset=train_dataset, data_collator=collator, optimizer=optimizer
)

generation_kwargs = {
    "max_new_tokens": script_args.max_new_tokens,
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0, 
    "do_sample": True,
    "temperature": 0.7,
    "pad_token_id": tokenizer.eos_token_id,
    "begin_suppress_tokens": [tokenizer.eos_token_id] ,
}


def _clean_responses(decoded_responses):
    cleaned = []
    for response in decoded_responses:
        response = response.strip('[PAD] ')
        response = response.strip('<unk>')
        temp_resp = response.strip('<s>').strip('</s>')
        temp_resp = temp_resp.split('\n\nHuman:')[0].strip()
        temp_resp = temp_resp.split('\nHuman:')[0].strip()
        temp_resp = temp_resp.split('\n\nAssistant:')[0].strip()
        temp_resp = temp_resp.split('\nAssistant:')[0].strip()
        temp_resp = temp_resp.split('\n\n\n')[0].strip()
        temp_resp = temp_resp.split('###')[0].strip()
        cleaned.append(temp_resp)
    return cleaned


def _compute_scalar_rewards(queries, responses):
    texts_merge = [q + r for q, r in zip(queries, responses)]
    queries_responses = [
        (instructions.get_input(text), instructions.get_response(text))
        for text in texts_merge
    ]

    if hasattr(instructions, 'get_post'):
        rewards_list = reward_model.get_reward_model_scores(queries_responses, instructions.get_post)
    else:
        rewards_list = reward_model.get_reward_model_scores(queries_responses)

    scalar_rewards = []
    for j in range(len(queries_responses)):
        scalar_reward = 0.0
        for k in range(num_rewards):
            signed_score = reward_sign_list[k] * float(rewards_list[k][j])
            scalar_reward += preference[k] * signed_score
        scalar_rewards.append(round(scalar_reward, 4))

    return rewards_list, scalar_rewards


@torch.no_grad()
def run_periodic_validation(global_step):
    if script_args.eval_steps is None or script_args.eval_steps <= 0:
        return
    if len(eval_dataset) == 0:
        return

    # Avoid duplicated W&B tables in distributed training.
    if not accelerator.is_local_main_process:
        return

    n_eval = min(script_args.eval_num_prompts, len(eval_dataset))
    eval_batch = [eval_dataset[idx] for idx in range(n_eval)]
    eval_queries = [ex["query"] for ex in eval_batch]
    eval_query_tensors = [
        torch.as_tensor(ex["input_ids"], dtype=torch.long, device=gpu_id)
        for ex in eval_batch
    ]

    was_training = model.training
    model.eval()
    model.gradient_checkpointing_disable()
    model.pretrained_model.config.use_cache = True

    eval_generation_kwargs = dict(generation_kwargs)
    eval_generation_kwargs["max_new_tokens"] = script_args.eval_max_new_tokens

    response_tensors = ppo_trainer.generate(
        eval_query_tensors,
        return_prompt=False,
        **eval_generation_kwargs,
    )
    decoded = tokenizer.batch_decode(response_tensors)
    eval_responses = _clean_responses(decoded)

    rewards_list, scalar_rewards = _compute_scalar_rewards(eval_queries, eval_responses)

    columns = ["step", "idx", "prompt", "response"]
    for name in reward_names:
        columns.append(f"{name}_raw")
        columns.append(f"{name}_signed")
    columns.append("scalar_reward")

    table = wandb.Table(columns=columns)
    for row_idx, (query, response, scalar_reward) in enumerate(zip(eval_queries, eval_responses, scalar_rewards)):
        row = [global_step, row_idx, query, response]
        for k, name in enumerate(reward_names):
            raw_score = float(rewards_list[k][row_idx])
            row.append(raw_score)
            row.append(float(reward_sign_list[k]) * raw_score)
        row.append(float(scalar_reward))
        table.add_data(*row)

    if not script_args.disable_wandb:
        if wandb.run is None:
            wandb.init(
                project=os.environ.get("WANDB_PROJECT", "pcma"),
                entity=os.environ.get("WANDB_ENTITY", None),
                name=script_args.wandb_name,
            )
        wandb.log({
            f"validation_samples/step_{global_step}": table,
            "validation/scalar_reward_mean": float(np.mean(scalar_rewards)),
            "validation/scalar_reward_std": float(np.std(scalar_rewards)),
        }, step=global_step)

    print(f"[eval] step={global_step}, logged {n_eval} validation generations to wandb", flush=True)

    if was_training:
        model.train()
    model.pretrained_model.config.use_cache = False


print("Training........")
model.gradient_checkpointing_disable()
model.pretrained_model.config.use_cache = True
epochs = script_args.epochs
mean_scores = []
std_scores = []
save_data = {
    'kl_mean': [],
    'reward_mean': [],
    'reward_std': [],
    'text_sample':[],
    'batch_time':[],
    'total_time':[],
}
t_start = time.time()
global_step = 0
for epoch in range(epochs):
    pbar = tqdm(total=len(train_dataset) // script_args.batch_size // accelerator.num_processes)
    for i, batch in enumerate(ppo_trainer.dataloader):
        t_epoch_start = time.time()
        print('epoch {}, batch {}'.format(epoch, i))
        query_tensors = [
            torch.as_tensor(q, dtype=torch.long, device=gpu_id)
            for q in batch["input_ids"]
        ]

        model.gradient_checkpointing_disable()
        model.pretrained_model.config.use_cache = True
            
        with torch.no_grad():
            response_tensors = ppo_trainer.generate(query_tensors, return_prompt=False, **generation_kwargs)

        full_responses = tokenizer.batch_decode(response_tensors)
        clean_texts = _clean_responses(full_responses)
        clean_response_tensors = [tokenizer.encode(text) for text in clean_texts]
        
        lengths = [len(clean_response_tensors[j]) for j in range(len(clean_response_tensors))]
        response_tensors = [response_tensors[j][:np.max([lengths[j], 2])] for j in range(len(response_tensors))]
        batch['response'] = clean_texts

        # Compute score
        texts_merge = [q + r for q, r in zip(batch['query'], batch['response'])]
        rewards_list, rewards = _compute_scalar_rewards(batch['query'], batch['response'])
        rewards_tensor = [torch.tensor(r).to(gpu_id) for r in rewards]
        print("iter {}, batch {}, mean score: {}".format(epoch, i, torch.mean(torch.tensor(rewards)).item()))

        model.gradient_checkpointing_enable()
        model.pretrained_model.config.use_cache = False
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards_tensor)
        policy_kl = [stats["objective/kl"]]
        ppo_trainer.log_stats(stats, batch, rewards)

        global_step += 1
        if script_args.eval_steps > 0 and global_step % script_args.eval_steps == 0:
            run_periodic_validation(global_step)
        accelerator.wait_for_everyone()

        all_rewards = accelerator.gather_for_metrics(rewards)
        all_policy_kl = accelerator.gather_for_metrics(policy_kl)
        if process_id == 0:
            mean_scores.append(torch.mean(torch.tensor(all_rewards)).item())
            std_scores.append(torch.std(torch.tensor(all_rewards)).item())
            save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'scores.png')
            plt.plot(mean_scores)
            plt.fill_between(np.arange(len(mean_scores)), np.array(mean_scores)- np.array(std_scores), np.array(mean_scores) + np.array(std_scores), alpha=0.5)
            plt.savefig(save_path)
            t_epoch_end = time.time()
            save_data['batch_time'].append(t_epoch_end - t_epoch_start)
            save_data['total_time'].append(t_epoch_end - t_start)
            save_data['kl_mean'].append(np.mean(all_policy_kl))
            save_data['reward_mean'] = mean_scores
            save_data['reward_std'] = std_scores
            save_data['text_sample'].append(texts_merge[0])
            dataframe = pd.DataFrame(save_data)
            dataframe.to_csv(os.path.join(script_args.save_directory, script_args.wandb_name,'data.csv'))
            print("iter {}, batch {}: log finish".format(epoch, i))

        # wait for the main process
        accelerator.wait_for_everyone()
        pbar.update(1)

        # save model
        if ppo_trainer.accelerator.is_main_process and i % 100 == 0 and i != 0:
            save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'batch_{}'.format(i))
            ppo_trainer.save_pretrained(save_path)
            print("iter {}, batch {}: model saved".format(epoch, i))
    
    # save model
    if ppo_trainer.accelerator.is_main_process:
        save_path = os.path.join(script_args.save_directory, script_args.wandb_name, 'batch_{}'.format(i))
        ppo_trainer.save_pretrained(save_path)
        print("iter {}, batch {}: model saved".format(epoch, i))
        