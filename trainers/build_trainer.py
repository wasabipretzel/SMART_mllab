from trainers.instructblip_trainer import InstructblipTrainer
from transformers import Seq2SeqTrainer


def get_trainer(model_args, training_args, model, metric, processor, data_module):
    if model_args.model_type=="instructblip":
        trainer = InstructblipTrainer(
                    model=model,
                    args=training_args,
                    compute_metrics=metric.compute_metrics,
                    tokenizer=processor.tokenizer if model_args.model_type=="instructblip" else None, #for prediction
                    **data_module
                )
    elif model_args.model_type=="R50_BERT":
        trainer = Seq2SeqTrainer(
                    model=model,
                    args=training_args,
                    compute_metrics=metric.compute_metrics,
                    **data_module
                )
    else:
        raise NotImplementedError
    return trainer
