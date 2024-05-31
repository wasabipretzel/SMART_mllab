"""_summary_
    trainer for training instructblip base model. Specifically, purpose of this trainer is for applying 
    different learning rates for pretrained/scratch modules in model
"""
import numpy as np
from transformers import Seq2SeqTrainer
import torch 
from torch import nn
from torch.utils.data import Dataset, DataLoader
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from transformers.utils import logging
from transformers.trainer_utils import has_length, EvalLoopOutput, EvalPrediction, PredictionOutput, denumpify_detensorize
from transformers.trainer_pt_utils import EvalLoopContainer, find_batch_size, is_torch_xla_available
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

logger = logging.get_logger(__name__)

class InstructblipTrainer(Seq2SeqTrainer):

    def create_optimizer(self):
        """
        Setup the optimizer.

        In instructblip base model, 
        pretrained module parts are ["qformer", "language_projection"] (need to use very small lr)
        scratch module parts are ["llm lora", "qformer lora"(not implemented), "vit adapter"(not implemented)] (need to use relatively large lr)

        #TODO : 나중에 qformer에 lora붙히고 난 이후에 lora제외한 qformer와 나머지 qformer을 분리할 수 있도록 디자인해야함
                => vision adapter도 마찬가지
        #TODO : pretrained_parameters, scratch_parameters에서 bias은 어떻게 처리해주는게 맞지?
        """ 
        opt_model = self.model
        pretrained_modules = ["qformer", "language_projection"]
        scratch_modules = ["lora", "sam_linear", "category_cls_head"]

        if self.args.pretrained_module_lr is not None and self.args.scratch_module_lr is not None: 
            pretrained_parameters, scratch_parameters= list(), list()
            for name, param in opt_model.named_parameters():
                for module in pretrained_modules:
                    if module in name and param.requires_grad:
                        pretrained_parameters.append(param)
                for module in scratch_modules:
                    if module in name and param.requires_grad:
                        scratch_parameters.append(param)

            optimizer_grouped_parameters = [
                {'params': pretrained_parameters, 'lr': self.args.pretrained_module_lr},
                {'params': scratch_parameters, 'lr': self.args.scratch_module_lr}
            ]

        optimizer_cls, optimizer_kwargs = Seq2SeqTrainer.get_optimizer_cls_and_kwargs(self.args)

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)


        return self.optimizer


    def log(self, logs:Dict[str, float]):
        """
            logs : {'loss': 4.2812, 'learning_rate': 7.042253521126761e-11}
            -> learning_rate을 빼고 pretrained_lr, scratch_lr로 보고하도록..
        """
        if 'eval' in list(logs.keys())[0]:
            super().log(logs)
        else:
            key_names = ["pretrained_lr", "scratch_lr"]
            step = logs.get("step", 0)
            if "learning_rate" in logs.keys():
                del logs["learning_rate"]
            lr_dict = {
                f'{key_names[i]}': group['lr'] for i, group in enumerate(self.optimizer.param_groups)
            }
            logs.update(lr_dict)
            super().log(logs)

class InstructblipTrainerCLS(Seq2SeqTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.category_loss = None

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            if model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        
        #============custom multiple loss =======================
        cls_loss = outputs["category_loss"]
        if cls_loss is not None:
            loss += cls_loss
        
        # Log the separate losses
        # self.log({'main_loss': outputs.loss.item()})
        if cls_loss is not None:
            self.category_loss = cls_loss.item()
            # self.log({'cls_loss': cls_loss.item()})

        return (loss, outputs) if return_outputs else loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        In instructblip base model, 
        pretrained module parts are ["qformer", "language_projection"] (need to use very small lr)
        scratch module parts are ["llm lora", "qformer lora"(not implemented), "vit adapter"(not implemented)] (need to use relatively large lr)

        #TODO : 나중에 qformer에 lora붙히고 난 이후에 lora제외한 qformer와 나머지 qformer을 분리할 수 있도록 디자인해야함
                => vision adapter도 마찬가지
        #TODO : pretrained_parameters, scratch_parameters에서 bias은 어떻게 처리해주는게 맞지?
        """ 
        opt_model = self.model
        pretrained_modules = ["qformer", "language_projection"]
        scratch_modules = ["lora", "sam_linear", "category_cls_head"]

        if self.args.pretrained_module_lr is not None and self.args.scratch_module_lr is not None: 
            pretrained_parameters, scratch_parameters= list(), list()
            for name, param in opt_model.named_parameters():
                for module in pretrained_modules:
                    if module in name and param.requires_grad:
                        pretrained_parameters.append(param)
                for module in scratch_modules:
                    if module in name and param.requires_grad:
                        scratch_parameters.append(param)

            optimizer_grouped_parameters = [
                {'params': pretrained_parameters, 'lr': self.args.pretrained_module_lr},
                {'params': scratch_parameters, 'lr': self.args.scratch_module_lr}
            ]

        optimizer_cls, optimizer_kwargs = Seq2SeqTrainer.get_optimizer_cls_and_kwargs(self.args)

        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)


        return self.optimizer


    def log(self, logs:Dict[str, float]):
        """
            logs : {'loss': 4.2812, 'learning_rate': 7.042253521126761e-11}
            -> learning_rate을 빼고 pretrained_lr, scratch_lr로 보고하도록..
        """
        if 'eval' in list(logs.keys())[0]:
            super().log(logs)
        else:
            key_names = ["pretrained_lr", "scratch_lr"]
            step = logs.get("step", 0)
            if "learning_rate" in logs.keys():
                del logs["learning_rate"]
            lr_dict = {
                f'{key_names[i]}': group['lr'] for i, group in enumerate(self.optimizer.param_groups)
            }
            if self.category_loss != None:
                lr_dict["category_loss"] = self.category_loss
            logs.update(lr_dict)
            super().log(logs)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ):
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args
        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        # if eval is called w/o train, handle model prep here
        if self.is_deepspeed_enabled and self.deepspeed is None:
            _, _ = deepspeed_init(self, num_training_steps=0, inference=True)

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        all_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_category_losses = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_preds = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_labels = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)
        all_inputs = EvalLoopContainer(self.args.eval_do_concat_batches, padding_index=-100)

        # Will be useful when we have an iterable dataset so don't know its length.
        observed_num_examples = 0

        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, category_loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(self.model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            if is_torch_xla_available():
                xm.mark_step()

            # Update containers
            if loss is not None:
                losses = self.gather_function((loss.repeat(batch_size)))
                all_losses.add(losses)
            if category_loss is not None:
                category_losses = self.gather_function((category_loss.repeat(batch_size)))
                all_category_losses.add(category_losses)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.gather_function((inputs_decode))
                all_inputs.add(inputs_decode)
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.gather_function((logits))
                all_preds.add(logits)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
                labels = self.gather_function((labels))
                all_labels.add(labels)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                all_losses.to_cpu_and_numpy()
                all_category_losses.to_cpu_and_numpy()
                all_preds.to_cpu_and_numpy()
                all_labels.to_cpu_and_numpy()
                all_inputs.to_cpu_and_numpy()

        # After all calls to `.gather_function`, reset to `gather_for_metrics`:
        self.gather_function = self.accelerator.gather_for_metrics
        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        all_losses = all_losses.get_arrays()
        all_category_losses = all_category_losses.get_arrays()
        all_preds = all_preds.get_arrays()
        all_labels = all_labels.get_arrays()
        all_inputs = all_inputs.get_arrays()

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)
        if isinstance(all_losses, list) and all_losses:
            metrics[f"{metric_key_prefix}_loss"] = np.concatenate(all_losses).mean().item()
        elif isinstance(all_losses, np.ndarray):
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()

        if isinstance(all_category_losses, list) and all_category_losses and category_loss is not None:
            metrics[f"{metric_key_prefix}_category_loss"] = np.concatenate(all_category_losses).mean().item()
        elif isinstance(all_category_losses, np.ndarray) and category_loss is not None:
            metrics[f"{metric_key_prefix}_category_loss"] = all_category_losses.mean().item()

        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
        **gen_kwargs,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            gen_kwargs:
                Additional `generate` specific kwargs.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only=prediction_loss_only, ignore_keys=ignore_keys
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        # Priority (handled in generate):
        # non-`None` gen_kwargs > model.generation_config > default GenerationConfig()
        if len(gen_kwargs) == 0 and hasattr(self, "_gen_kwargs"):
            gen_kwargs = self._gen_kwargs.copy()
        if "num_beams" in gen_kwargs and gen_kwargs["num_beams"] is None:
            gen_kwargs.pop("num_beams")
        if "max_length" in gen_kwargs and gen_kwargs["max_length"] is None:
            gen_kwargs.pop("max_length")

        default_synced_gpus = True if is_deepspeed_zero3_enabled() else False
        gen_kwargs["synced_gpus"] = (
            gen_kwargs["synced_gpus"] if gen_kwargs.get("synced_gpus") is not None else default_synced_gpus
        )

        generation_inputs = inputs.copy()
        # If the `decoder_input_ids` was created from `labels`, evict the former, so that the model can freely generate
        # (otherwise, it would continue generating from the padded `decoder_input_ids`)
        if (
            "labels" in generation_inputs
            and "decoder_input_ids" in generation_inputs
            and generation_inputs["labels"].shape == generation_inputs["decoder_input_ids"].shape
        ):
            generation_inputs = {
                k: v for k, v in inputs.items() if k not in ("decoder_input_ids", "decoder_attention_mask")
            }
        generated_tokens = self.model.generate(**generation_inputs, **gen_kwargs)

        # Temporary hack to ensure the generation config is not initialized for each iteration of the evaluation loop
        # TODO: remove this hack when the legacy code that initializes generation_config from a model config is
        # removed in https://github.com/huggingface/transformers/blob/98d88b23f54e5a23e741833f1e973fdf600cc2c5/src/transformers/generation/utils.py#L1183
        if self.model.generation_config._from_model_config:
            self.model.generation_config._from_model_config = False

        # Retrieves GenerationConfig from model.generation_config
        gen_config = self.model.generation_config
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_config.max_length:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_length)
        elif gen_config.max_new_tokens is not None and generated_tokens.shape[-1] < gen_config.max_new_tokens + 1:
            generated_tokens = self._pad_tensors_to_max_len(generated_tokens, gen_config.max_new_tokens + 1)

        with torch.no_grad():
            if has_labels:
                with self.compute_loss_context_manager():
                    outputs = model(**inputs)
                if self.label_smoother is not None:
                    loss = self.label_smoother(outputs, inputs["labels"]).mean().detach()
                else:
                    loss = (outputs["loss"] if isinstance(outputs, dict) else outputs[0]).mean().detach()
                    if outputs["category_loss"] != None:
                        category_loss = (outputs["category_loss"]).mean().detach()
                    else:
                        category_loss = None
            else:
                loss = None

        if self.args.prediction_loss_only:
            return loss, None, None

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_config.max_length:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_length)
            elif gen_config.max_new_tokens is not None and labels.shape[-1] < gen_config.max_new_tokens + 1:
                labels = self._pad_tensors_to_max_len(labels, gen_config.max_new_tokens + 1)
        else:
            labels = None