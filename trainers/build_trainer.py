from trainers.flant5_trainer import T5Trainer
from transformers import Seq2SeqTrainer


def get_trainer(model_args, training_args, model, metric, tokenizer, data_module):
    if "flant5" in model_args.llm_model_type:
        trainer = T5Trainer(
                    model=model,
                    args=training_args,
                    compute_metrics=metric.compute_metrics,
                    tokenizer=tokenizer,
                    **data_module
                )
    else:
        raise NotImplementedError

    return trainer