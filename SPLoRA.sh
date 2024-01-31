#!/bin/bash
pretrained_model="/models/Stable-diffusion/v1-5-pruned.safetensors" 
is_v2_model=0                             
parameterization=0                        
train_data_dir="./train/image_pokemon"              
reg_data_dir=""                           
network_module="networks.lora" 
network_weights=""             
network_dim=64                 
network_alpha=32               
resolution="512,512"  
batch_size=1         
max_train_epoches=400  
save_every_n_epochs=50 
train_unet_only=0       
train_text_encoder_only=0 
stop_text_encoder_training=0 
noise_offset="0"  
keep_tokens=0   
min_snr_gamma=0 
lr="1e-5"
unet_lr="1e-5"
text_encoder_lr="1e-5"
lr_scheduler="constant" 
lr_warmup_steps=0                   
lr_restart_cycles=1                 
output_name="pokemon_lora_mia_stable_alpha005_analysis_v3_1e5_"           
save_model_as="safetensors" 
save_state=0 
resume=""    
min_bucket_reso=256              
max_bucket_reso=1024             
persistent_data_loader_workers=0 
clip_skip=2                     
multi_gpu=0 
lowram=0 
optimizer_type="AdamW8bit" 
algo="lora"  
conv_dim=4   
conv_alpha=4 
dropout="0"  
use_wandb=0 
wandb_api_key="" 
log_tracker_name="" 


export HF_HOME="huggingface"
export TF_CPP_MIN_LOG_LEVEL=3

extArgs=()
launchArgs=()
if [[ $multi_gpu == 1 ]]; then launchArgs+=("--multi_gpu"); fi

if [[ $is_v2_model == 1 ]]; then
  extArgs+=("--v2");
else
  extArgs+=("--clip_skip $clip_skip");
fi

if [[ $parameterization == 1 ]]; then extArgs+=("--v_parameterization"); fi

if [[ $train_unet_only == 1 ]]; then extArgs+=("--network_train_unet_only"); fi

if [[ $train_text_encoder_only == 1 ]]; then extArgs+=("--network_train_text_encoder_only"); fi

if [[ $network_weights ]]; then extArgs+=("--network_weights $network_weights"); fi

if [[ $reg_data_dir ]]; then extArgs+=("--reg_data_dir $reg_data_dir"); fi

if [[ $optimizer_type ]]; then extArgs+=("--optimizer_type $optimizer_type"); fi

if [[ $optimizer_type == "DAdaptation" ]]; then extArgs+=("--optimizer_args decouple=True"); fi

if [[ $optimizer_type == "DAdaptAdam" ]]; then extArgs+=("--optimizer_args decouple=True"); fi

if [[ $optimizer_type == "Prodigy" ]]; then extArgs+=("--optimizer_args safeguard_warmup=True use_bias_correction=True weight_decay=0.01"); fi

if [[ $save_state == 1 ]]; then extArgs+=("--save_state"); fi

if [[ $resume ]]; then extArgs+=("--resume $resume"); fi

if [[ $persistent_data_loader_workers == 1 ]]; then extArgs+=("--persistent_data_loader_workers"); fi

if [[ $network_module == "lycoris.kohya" ]]; then
  extArgs+=("--network_args conv_dim=$conv_dim conv_alpha=$conv_alpha algo=$algo dropout=$dropout")
fi

if [[ $stop_text_encoder_training -ne 0 ]]; then extArgs+=("--stop_text_encoder_training $stop_text_encoder_training"); fi

if [[ $noise_offset != "0" ]]; then extArgs+=("--noise_offset $noise_offset"); fi

if [[ $min_snr_gamma -ne 0 ]]; then extArgs+=("--min_snr_gamma $min_snr_gamma"); fi

if [[ $use_wandb == 1 ]]; then
  extArgs+=("--log_with=all")
else
  extArgs+=("--log_with=tensorboard")
fi

if [[ $wandb_api_key ]]; then extArgs+=("--wandb_api_key $wandb_api_key"); fi

if [[ $log_tracker_name ]]; then extArgs+=("--log_tracker_name $log_tracker_name"); fi

if [[ $lowram ]]; then extArgs+=("--lowram"); fi

python -m accelerate.commands.launch ${launchArgs[@]} --num_cpu_threads_per_process=8 "./sd-scripts/splora.py" \
  --enable_bucket \
  --pretrained_model_name_or_path=$pretrained_model \
  --train_data_dir=$train_data_dir \
  --output_dir="./output" \
  --logging_dir="./logs" \
  --log_prefix=$output_name \
  --resolution=$resolution \
  --network_module=$network_module \
  --max_train_epochs=$max_train_epoches \
  --learning_rate=$lr \
  --unet_lr=$unet_lr \
  --text_encoder_lr=$text_encoder_lr \
  --lr_scheduler=$lr_scheduler \
  --lr_warmup_steps=$lr_warmup_steps \
  --lr_scheduler_num_cycles=$lr_restart_cycles \
  --network_dim=$network_dim \
  --network_alpha=$network_alpha \
  --output_name=$output_name \
  --train_batch_size=$batch_size \
  --save_every_n_epochs=$save_every_n_epochs \
  --mixed_precision="fp16" \
  --save_precision="fp16" \
  --seed="13" \
  --cache_latents \
  --prior_loss_weight=1 \
  --max_token_length=225 \
  --caption_extension=".txt" \
  --save_model_as=$save_model_as \
  --min_bucket_reso=$min_bucket_reso \
  --max_bucket_reso=$max_bucket_reso \
  --keep_tokens=$keep_tokens \
  --xformers --shuffle_caption ${extArgs[@]} \
  --k_steps=1 \
  --param_alpha=1 \
  --param_beta=1 \
