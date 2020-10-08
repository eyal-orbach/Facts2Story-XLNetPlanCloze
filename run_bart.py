import argparse
import glob
import logging
import os
import time

import torch
from torch.utils.data import DataLoader

from datareaders.MaskedPlotFactEmbeddingDatasetBart import MaskedPlotFactEmbeddingDatasetBart
from transformer_base import BaseTransformer, add_generic_args, generic_train, get_linear_schedule_with_warmup
from transformers import BartTokenizer
#from pytorch_lightning.loggers import TensorBoardLogger

logger = logging.getLogger(__name__)

results_path = "/home/nlp/eyalo/expirements/salieplots/bart/emnlp_redo_x"
def print_generation_results(inputs, preds, target, batch_idx):
    path = results_path + "/batch"+str(batch_idx) +"/"
    os.makedirs(path, exist_ok=True)
    for i in range(len(preds)):
        with open(path+"pred"+str(i)+".txt", "w", encoding='utf-8') as f:
            f.write("generated text:\n")
            f.write(preds[i])
            f.write("\n-------\noriginal text:\n")
            f.write(target[i])
            f.write("\n-------\ninputs text:\n")
            f.write(inputs[i])



class BartSystem(BaseTransformer):

    mode = "language-modeling"
    tokenizer2 = None

    def __init__(self, hparams):
        super(BartSystem, self).__init__(hparams, num_labels=None, mode=self.mode)
        self.tokenizer2 = BartTokenizer.from_pretrained("bart-large")

    def forward(
        self, input_ids, attention_mask=None, decoder_input_ids=None, decoder_attention_mask=None, lm_labels=None
    ):
        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            lm_labels=lm_labels,
        )

    def _step(self, batch):
        y = batch["target_ids"]
        y_ids = y[:, :-1].contiguous()
        lm_labels = y[:, 1:].clone()
        lm_labels[y[:, 1:] == self.tokenizer.pad_token_id] = -100
        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            decoder_input_ids=y_ids,
            lm_labels=lm_labels,
        )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        print("train loss: " + str(loss))
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        print("validation loss: " + str(avg_loss))
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def test_step(self, batch, batch_idx):
        generated_ids = self.model.generate(
            batch["source_ids"],
            attention_mask=batch["source_mask"],
            num_beams=1,
            do_sample=True,
            top_p=0.9,
            max_length=800,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
        )


        preds = [
            self.tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for g in generated_ids
        ]
        target = [
            self.tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for t in batch["target_ids"]
        ]

        inputs = [
            self.tokenizer.decode(s, skip_special_tokens=False, clean_up_tokenization_spaces=True)
            for s in batch["source_ids"]
        ]

        loss = self._step(batch)
        print_generation_results(inputs, preds, target, batch_idx)

        return {"val_loss": loss, "preds": preds, "target": target}

    def test_end(self, outputs):
        return self.validation_end(outputs)

    def test_epoch_end(self, outputs):
        # output_test_predictions_file = os.path.join(self.hparams.output_dir, "test_predictions.txt")
        # output_test_targets_file = os.path.join(self.hparams.output_dir, "test_targets.txt")
        output_test_predictions_file = os.path.join(results_path, "test_predictions.txt")
        output_test_targets_file = os.path.join(results_path, "test_targets.txt")
        # write predictions and targets for later rouge evaluation.
        with open(output_test_predictions_file, "w+", encoding='utf8') as p_writer, open(output_test_targets_file, "w+", encoding='utf8') as t_writer:
            for output_batch in outputs:
                p_writer.writelines(s + "\n" for s in output_batch["preds"])
                t_writer.writelines(s + "\n" for s in output_batch["target"])
            p_writer.close()
            t_writer.close()

        return self.test_end(outputs)

    def train_dataloader(self):
        train_dataset = MaskedPlotFactEmbeddingDatasetBart(
            self.tokenizer, args, self.hparams.data_dir +"train/", block_size=self.hparams.max_seq_length, is_train=False
        )
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, self.hparams.n_gpu)))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = MaskedPlotFactEmbeddingDatasetBart(
            self.tokenizer, args, self.hparams.data_dir +"valid/", block_size=self.hparams.max_seq_length, is_train=False
        )
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size)

    def test_dataloader(self):
        test_dataset = MaskedPlotFactEmbeddingDatasetBart(
            self.tokenizer, args, self.hparams.data_dir +"test/", block_size=self.hparams.max_seq_length, is_train=False
        )
        # return DataLoader(test_dataset, batch_size=self.hparams.eval_batch_size)
        return DataLoader(test_dataset, batch_size=args.eval_batch_size)

    @staticmethod
    def add_model_specific_args(parser, root_dir):
        BaseTransformer.add_model_specific_args(parser, root_dir)
        # Add BART specific options
        parser.add_argument("--set_gpu", type=int, default=[], action='append')
        parser.add_argument(
            "--max_seq_length",
            default=1024,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded.",
        )

        parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the dataset files for the CNN/DM summarization task.",
        )

        parser.add_argument(
            "--overwrite_cache", default=False, type=bool,
        )

        return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_generic_args(parser, os.getcwd())
    parser = BartSystem.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()
    #logger = TensorBoardLogger("tb_logs", name="my_model")

    # If output_dir not provided, a folder will be generated in pwd
    if args.output_dir is None:
        args.output_dir = os.path.join("./results", f"{args.task}_{args.model_type}_{time.strftime('%Y%m%d_%H%M%S')}",)
        os.makedirs(args.output_dir)

    if args.do_train:
        model = BartSystem(args)
        trainer = generic_train(model, args)

    # Optionally, predict on dev set and write to output_dir
    if args.do_predict:

        mydir = "/home/nlp/eyalo/bartout/bartoutput/"

        checkpoints = list(sorted(glob.glob(os.path.join(args.output_dir, "checkpoint*.ckpt"), recursive=True)))
        # checkpoints = list(sorted(glob.glob(os.path.join(mydir, "checkpointepoch=*.ckpt"), recursive=True)))
        model = BartSystem.load_from_checkpoint(checkpoints[-1])
        model.hparams.data_dir = args.data_dir
        trainer = generic_train(model, args)
        trainer.test(model)
