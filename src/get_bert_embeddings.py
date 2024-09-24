import pandas as pd
import torch
from transformers import BertTokenizer, BertModel
from config import Config
import math


class BertEmbeddingExtractor:
    def __init__(self, model_name="bert-base-cased", batch_size=16):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.batch_size = batch_size

    @torch.no_grad()
    def infer(self, inputs):
        self.model.eval()
        outputs = self.model(**inputs)
        return outputs

    def get_embeddings_in_batches(self, sentences):
        all_embeddings = []
        num_batches = math.ceil(len(sentences) / self.batch_size)

        for i in range(num_batches):
            batch_sentences = sentences[i * self.batch_size : (i + 1) * self.batch_size]

            inputs = self.tokenizer(
                batch_sentences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )

            outputs = self.infer(inputs)

            # get [CLS] token's embedding
            batch_embeddings = outputs.last_hidden_state[:, 0, :]
            all_embeddings.append(batch_embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)

        return all_embeddings


class EmbeddingProcessor:
    def __init__(
        self,
        extractor,
        input_folder="results/bert_inputs",
        output_folder="results/bert_embeddings",
    ):
        self.extractor = extractor
        self.input_folder = input_folder
        self.output_folder = output_folder

    def process_fold(self, fold_num):
        input_path = f"{self.input_folder}/bert_input_fold{fold_num}.csv"
        output_path = f"{self.output_folder}/bert_embeddings_fold{fold_num}.csv"

        fold_df = pd.read_csv(input_path)

        sentences = fold_df["sentence"].tolist()

        embeddings = self.extractor.get_embeddings_in_batches(sentences)

        embeddings_list = embeddings.tolist()

        embedding_df = pd.DataFrame(
            {
                "idx": fold_df["idx"],
                "raw_embeddings": embeddings_list,
            }
        )

        embedding_df.to_csv(output_path, index=False)
        print(f"Processed fold {fold_num} and saved embeddings to {output_path}")

    def process_all_folds(self, num_folds):
        for fold_num in range(1, num_folds + 1):
            self.process_fold(fold_num)


if __name__ == "__main__":
    extractor = BertEmbeddingExtractor(batch_size=16)

    processor = EmbeddingProcessor(extractor)

    processor.process_all_folds(Config.num_folds)
