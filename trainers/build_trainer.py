from trainers.instructblip_trainer import InstructblipTrainer
from trainers.idefics2_trainer import idefics2_trainer
from transformers import Seq2SeqTrainer


def get_trainer(model_args, training_args, model, metric, processor, data_module):
    if "instructblip" in model_args.model_type:
        trainer = InstructblipTrainer(
                    model=model,
                    args=training_args,
                    compute_metrics=metric.compute_metrics,
                    tokenizer=processor.tokenizer,
                    **data_module
                )
    elif 'idefics2' in model_args.pretrained_model_path:
        trainer = idefics2_trainer(
            model = model,
            args = training_args,
            compute_metrics = metric.compute_metrics,
            tokenizer = processor.tokenizer,
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
