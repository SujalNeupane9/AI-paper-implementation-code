
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DebertaV2Tokenizer
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
import torch
from torch.nn.functional import softmax
import logging
import os
from typing import Tuple, Dict
from datasets import load_dataset

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CFG:
    """Configuration class for training parameters"""
    model_name = "FacebookAI/xlm-roberta-large"
    epochs = 10
    learning_rate = 0.00003
    batch_size = 32
    max_length = 128
    test_size = 0.25
    output_dir = "./model"
    checkpoint_dir = "./checkpoints"
    random_seed = 42

class TextClassificationTrainer:
    def __init__(self):
        self.tokenizer = DebertaV2Tokenizer.from_pretrained(
            CFG.model_name,
            add_prefix_space=True  # This can help with better tokenization
        )
        self.setup_directories()

    @staticmethod
    def setup_directories():
        """Create necessary directories for model and checkpoint saving"""
        os.makedirs(CFG.output_dir, exist_ok=True)
        os.makedirs(CFG.checkpoint_dir, exist_ok=True)

    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
      """Load training and test data from the Hugging Face datasets library"""
      try:
          dataset = load_dataset("Jinyan1/COLING_2025_MGT_en")

          # Ensure required splits exist
          if 'train' not in dataset or 'dev' not in dataset:
              raise ValueError("Dataset must have 'train' and 'test' splits.")

          # Convert to DataFrame
          train_df = dataset['train'].to_pandas()
          test_df = dataset['dev'].to_pandas()

          # Validate and rename columns as necessary
          required_train_cols = ['text', 'label']
          required_test_cols = ['text', 'label']

          missing_train_cols = [col for col in required_train_cols if col not in train_df.columns]
          missing_test_cols = [col for col in required_test_cols if col not in test_df.columns]

          if missing_train_cols:
              raise ValueError(f"Train split missing required columns: {missing_train_cols}")
          if missing_test_cols:
              raise ValueError(f"Test split missing required columns: {missing_test_cols}")

          # Rename 'labels' to 'label' in test set
          test_df = test_df.rename(columns={'labels': 'label'})

          return train_df, test_df
      except Exception as e:
          raise Exception(f"Error loading dataset: {e}")


    def preprocess_function(self, examples):
        """Tokenize text data"""
        return self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=CFG.max_length,
            padding=True
        )

    def preprocess_data(self, train: pd.DataFrame, test: pd.DataFrame) -> Tuple[DatasetDict, Dataset]:
        """Convert DataFrames to Datasets and preprocess"""
        try:
            # Convert to Dataset format
            ds = Dataset.from_pandas(train)
            ds_test = Dataset.from_pandas(test)

            # Tokenize datasets
            tok_ds = ds.map(self.preprocess_function, batched=True)
            dds = tok_ds.train_test_split(test_size=CFG.test_size, seed=CFG.random_seed)
            eval_dataset = ds_test.map(self.preprocess_function, batched=True)

            return dds, eval_dataset
        except Exception as e:
            raise Exception(f"Error in data preprocessing: {e}")

    @staticmethod
    def compute_metrics(eval_pred) -> Dict[str, float]:
        """Compute evaluation metrics"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)

        metrics = {}

        # Accuracy
        accuracy_metric = evaluate.load("accuracy")
        accuracy_result = accuracy_metric.compute(predictions=predictions, references=labels)
        metrics["accuracy"] = accuracy_result["accuracy"]

        # Precision
        precision_metric = evaluate.load("precision")
        precision_result = precision_metric.compute(
            predictions=predictions,
            references=labels,
            average="weighted",
            zero_division=0
        )
        metrics["precision"] = precision_result["precision"]

        # Recall
        recall_metric = evaluate.load("recall")
        recall_result = recall_metric.compute(
            predictions=predictions,
            references=labels,
            average="weighted",
            zero_division=0
        )
        metrics["recall"] = recall_result["recall"]

        # F1 Score
        f1_metric = evaluate.load("f1")
        f1_result = f1_metric.compute(
            predictions=predictions,
            references=labels,
            average="weighted"
        )
        metrics["f1"] = f1_result["f1"]

        return metrics

    def train_model(self, dds: DatasetDict) -> Trainer:
        """Initialize and train the model"""
        try:
            # Initialize model and data collator
            model = AutoModelForSequenceClassification.from_pretrained(
                CFG.model_name,
                num_labels=2,ignore_mismatched_sizes=True
            )
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

            # Set up training arguments
            training_args = TrainingArguments(
                output_dir=CFG.output_dir,
                learning_rate=CFG.learning_rate,
                per_device_train_batch_size=CFG.batch_size,
                per_device_eval_batch_size=CFG.batch_size,
                num_train_epochs=CFG.epochs,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                fp16=torch.cuda.is_available(),
                logging_dir=f"{CFG.output_dir}/logs",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
            )

            # Initialize trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dds['train'],
                eval_dataset=dds['test'],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=self.compute_metrics,
            )

            # Train the model
            trainer.train()
            return trainer
        except Exception as e:
            raise Exception(f"Error in model training: {e}")

    def get_predictions(self, trainer: Trainer, eval_dataset: Dataset) -> pd.DataFrame:
        """Get probability predictions for the evaluation dataset and return as DataFrame"""
        try:
            predictions = trainer.predict(eval_dataset)
            logits = predictions.predictions
            probabilities = softmax(torch.tensor(logits), dim=-1).numpy()

            # Create DataFrame with model-specific column names
            model_name = CFG.model_name.split('/')[-1]  # Get last part of model name
            df_predictions = pd.DataFrame(
                probabilities,
                columns=[f'p0_{model_name}', f'p1_{model_name}']
            )

            return df_predictions

        except Exception as e:
            raise Exception(f"Error in getting predictions: {e}")

    def cleanup(self):
        """Clean up GPU memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    """Main execution function"""
    try:
        # Initialize trainer
        text_classifier = TextClassificationTrainer()

        # Load and preprocess data
        logger.info("Loading data...")
        train, test = text_classifier.load_data()

        logger.info("Preprocessing data...")
        dds, eval_dataset = text_classifier.preprocess_data(train, test)

        # Train model
        logger.info("Training model...")
        trainer = text_classifier.train_model(dds)

        # Get predictions as DataFrame
        logger.info("Getting predictions...")
        df_predictions = text_classifier.get_predictions(trainer, eval_dataset)

        # Save predictions DataFrame
        output_path = f"{CFG.output_dir}/predictions_{CFG.model_name.split('/')[-1]}.csv"
        df_predictions.to_csv(output_path, index=False)
        logger.info(f"Predictions saved to {output_path}")

        # Cleanup
        text_classifier.cleanup()

        return df_predictions

    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise

if __name__ == "__main__":
    main()
