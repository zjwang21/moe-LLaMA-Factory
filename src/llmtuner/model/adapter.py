import inspect
from typing import TYPE_CHECKING

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, MoeConfig, PeftConfig
from ..loramoe.peft import LoraConfig as LoraMoeConfig
from ..loramoe.peft import TaskType as LoraMoeTaskType
from ..loramoe.peft import get_peft_model as get_peft_model_loramoe
from transformers.integrations import is_deepspeed_zero3_enabled

from ..extras.logging import get_logger
from .utils import find_all_linear_modules


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

    from ..hparams import FinetuningArguments, ModelArguments


logger = get_logger(__name__)


def init_adapter(
    model: "PreTrainedModel", model_args: "ModelArguments", finetuning_args: "FinetuningArguments", is_trainable: bool
) -> "PreTrainedModel":
    r"""
    Initializes the adapters.

    Support full-parameter, freeze and LoRA training.

    Note that the trainable parameters must be cast to float32.
    """
    if finetuning_args.finetuning_type == "pro":
        pass
    elif finetuning_args.finetuning_type == "moe":
        logger.info("Fine-tuning method: MOE")
        if model_args.adapter_name_or_path is not None:
            logger.info("Resume training from moe adapter: {}.".format(model_args.adapter_name_or_path[0]))
            moe_config = PeftConfig.from_pretrained(model_args.adapter_name_or_path[0])
            moe_config.topk = finetuning_args.topk
            moe_config.use_polarization_loss = finetuning_args.use_polarization_loss
            moe_config.polarization_coef = finetuning_args.polarization_coef
            moe_config.polarization_func = finetuning_args.polarization_func
            moe_config.use_polarization_loss  =finetuning_args.use_polarization_loss
            moe_config.save_router_logits = finetuning_args.save_router_logits
            moe_config.router_aux_loss_coef = finetuning_args.router_aux_loss_coef
            moe_config.use_load_balancing_loss=finetuning_args.use_load_balancing_loss
            moe_config.save_all_params=finetuning_args.save_all_params
            moe_config.ce_loss_coef=finetuning_args.ce_loss_coef
            model = PeftModel.from_pretrained(model, 
                                              model_id=model_args.adapter_name_or_path[0], 
                                              config=moe_config, 
                                              is_trainable=True
                                              )
        else:
            checkpoint_to_resume = None
            if is_trainable and checkpoint_to_resume is None: # create new lora weights while training
                distance = finetuning_args.moe_every_k_layers
                if distance is not None:
                    layers = list(range(distance - 1, len(model.model.layers), distance))
                else:
                    layers = [int(k) for k in finetuning_args.layers_to_moe.split(",")]
                moe_config = MoeConfig(
                    task_type=TaskType.CAUSAL_LM,
                    inference_mode=False,
                    router_type=finetuning_args.moe_router_type,
                    num_experts=finetuning_args.moe_num_experts,
                    layers_to_transform=layers,
                    save_all_params=finetuning_args.save_all_params,
                    num_heads=finetuning_args.moe_num_heads,
                    topk=finetuning_args.topk,
                    use_polarization_loss=finetuning_args.use_polarization_loss,
                    polarization_coef=finetuning_args.polarization_coef,
                    polarization_func=finetuning_args.polarization_func,
                    save_router_logits=finetuning_args.save_router_logits,
                    router_aux_loss_coef=finetuning_args.router_aux_loss_coef,
                    use_load_balancing_loss=finetuning_args.use_load_balancing_loss,
                    ce_loss_coef=finetuning_args.ce_loss_coef
                )
                model = get_peft_model(model, moe_config)
                
        if finetuning_args.save_all_params:
            logger.info("Moe training and saving all params.")
            for n, p in model.named_parameters():
                p.requires_grad = True

        logger.info(moe_config)
        if finetuning_args.train_only_router:
            logger.info("Mark only the moe router trainable.")
            for n, p in model.named_parameters():
                if "router" in n:
                    if finetuning_args.init_moe_weights:
                        print("Init moe router params......")
                        p.data.normal_(mean=0.0, std=0.02)
                    p.requires_grad = True
                else:
                    p.requires_grad = False


    if (not is_trainable) and model_args.adapter_name_or_path is None:
        logger.info("Adapter is not found at evaluation, load the base model.")
        return model

    if finetuning_args.finetuning_type == "ffn" and is_trainable:
        logger.info("Fine-tuning method: ffn")
        model = model.float()
        for n, p in model.named_parameters():
            if "mlp" in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

    if finetuning_args.finetuning_type == "full" and is_trainable:
        logger.info("Fine-tuning method: Full")
        model = model.float()

    if finetuning_args.finetuning_type == "freeze" and is_trainable:
        logger.info("Fine-tuning method: Freeze")
        num_layers = (
            getattr(model.config, "num_hidden_layers", None)
            or getattr(model.config, "num_layers", None)
            or getattr(model.config, "n_layer", None)
        )
        if not num_layers:
            raise ValueError("Current model does not support freeze tuning.")

        if finetuning_args.num_layer_trainable > 0:  # fine-tuning the last n layers if num_layer_trainable > 0
            trainable_layer_ids = [num_layers - k - 1 for k in range(finetuning_args.num_layer_trainable)]
        else:  # fine-tuning the first n layers if num_layer_trainable < 0
            trainable_layer_ids = [k for k in range(-finetuning_args.num_layer_trainable)]  # noqa: C416

        trainable_layers = []
        for module_name in finetuning_args.name_module_trainable:
            for idx in trainable_layer_ids:
                trainable_layers.append("{:d}.{}".format(idx, module_name))

        for name, param in model.named_parameters():
            if not any(trainable_layer in name for trainable_layer in trainable_layers):
                param.requires_grad_(False)
            else:
                param.data = param.data.to(torch.float32)

    if finetuning_args.finetuning_type == "lora":
        if finetuning_args.loramoe:
            logger.info("Init new peft model") 
            target_modules = finetuning_args.lora_target # lora paras
            lora_rank = finetuning_args.lora_rank
            lora_dropout = finetuning_args.lora_dropout
            lora_alpha = finetuning_args.lora_alpha
            lora_nums = 8
            blc_alpha = 0.1
            blc_weight = 0.1
            logger.info(f"lora_rank: {lora_rank}")
            logger.info(f"lora_nums: {lora_nums}")
            logger.info(f"blc_alpha: {blc_alpha}")
            logger.info(f"blc_weight: {blc_weight}")
            peft_config = LoraMoeConfig(
                task_type=LoraMoeTaskType.CAUSAL_LM,
                target_modules=target_modules,
                inference_mode=False,
                r=lora_rank, 
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                lora_nums=lora_nums,
                blc_alpha=blc_alpha,
                blc_weight=blc_weight,
                )
            model = get_peft_model_loramoe(model, peft_config)
        else:
            logger.info("Fine-tuning method: LoRA")
            adapter_to_resume = None

            if model_args.adapter_name_or_path is not None:
                is_mergeable = True
                if getattr(model, "quantization_method", None):  # merge lora in quantized model is unstable
                    assert len(model_args.adapter_name_or_path) == 1, "Quantized model only accepts a single adapter."
                    is_mergeable = False

                if is_deepspeed_zero3_enabled():
                    assert len(model_args.adapter_name_or_path) == 1, "Cannot use multiple adapters in DeepSpeed ZeRO-3."
                    is_mergeable = False

                if (is_trainable and not finetuning_args.create_new_adapter) or (not is_mergeable):
                    adapter_to_merge = model_args.adapter_name_or_path[:-1]
                    adapter_to_resume = model_args.adapter_name_or_path[-1]
                else:
                    adapter_to_merge = model_args.adapter_name_or_path

                for adapter in adapter_to_merge:
                    model = PeftModel.from_pretrained(model, adapter)
                    model = model.merge_and_unload()

                if len(adapter_to_merge) > 0:
                    logger.info("Merged {} adapter(s).".format(len(adapter_to_merge)))

                if adapter_to_resume is not None:  # resume lora training
                    model = PeftModel.from_pretrained(model, adapter_to_resume, is_trainable=is_trainable)

            if is_trainable and adapter_to_resume is None:  # create new lora weights while training
                if len(finetuning_args.lora_target) == 1 and finetuning_args.lora_target[0] == "all":
                    target_modules = find_all_linear_modules(model)
                else:
                    target_modules = finetuning_args.lora_target

                peft_kwargs = {
                    "r": finetuning_args.lora_rank,
                    "target_modules": target_modules,
                    "lora_alpha": finetuning_args.lora_alpha,
                    "lora_dropout": finetuning_args.lora_dropout,
                }

                if model_args.use_unsloth:
                    from unsloth import FastLlamaModel, FastMistralModel  # type: ignore

                    unsloth_peft_kwargs = {"model": model, "max_seq_length": model_args.model_max_length}
                    if "loftq_config" in inspect.signature(FastLlamaModel.get_peft_model).parameters:
                        unsloth_peft_kwargs["loftq_config"] = {}

                    if getattr(model.config, "model_type", None) == "llama":
                        model = FastLlamaModel.get_peft_model(**peft_kwargs, **unsloth_peft_kwargs)
                    elif getattr(model.config, "model_type", None) == "mistral":
                        model = FastMistralModel.get_peft_model(**peft_kwargs, **unsloth_peft_kwargs)
                    else:
                        raise NotImplementedError

                else:
                    lora_config = LoraConfig(
                        task_type=TaskType.CAUSAL_LM,
                        inference_mode=False,
                        modules_to_save=finetuning_args.additional_target,
                        **peft_kwargs,
                    )
                    model = get_peft_model(model, lora_config)

        for param in filter(lambda p: p.requires_grad, model.parameters()):
            param.data = param.data.to(torch.bfloat16 if finetuning_args.lora_bf16_mode else torch.float32)

    if model_args.adapter_name_or_path is not None:
        logger.info("Loaded adapter(s): {}".format(",".join(model_args.adapter_name_or_path)))

    return model
