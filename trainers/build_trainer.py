from trainers.instructblip_trainer import InstructblipTrainer
from transformers import Seq2SeqTrainer


def get_trainer(model_args, training_args, model, metric, processor, data_module):
    if "instructblip" in model_args.model_type:
        if model_args.category_classification_loss:
            trainer = InstructblipTrainerCLS(
                        model=model,
                        args=training_args,
                        compute_metrics=metric.compute_metrics,
                        tokenizer=processor.tokenizer,
                        **data_module
                    )

        else:
            trainer = InstructblipTrainer(
                        model=model,
                        args=training_args,
                        compute_metrics=metric.compute_metrics,
                        tokenizer=processor.tokenizer,
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
