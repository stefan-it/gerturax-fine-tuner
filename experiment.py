import logging

from dataclasses import dataclass

from flair import set_seed

from flair.datasets import NER_GERMAN_GERMEVAL, GERMEVAL_2018_OFFENSIVE_LANGUAGE
from flair.embeddings import TransformerWordEmbeddings, TransformerDocumentEmbeddings
from flair.models import SequenceTagger, TextClassifier
from flair.trainers import ModelTrainer
from flair.trainers.plugins.loggers.tensorboard import TensorboardLogger

from pathlib import Path

from conll03_german_revised_dataset import CONLL_03_GERMAN_REVISED
from germeval14_no_wikipedia_dataset import NER_GERMEVAL_2014_NO_WIKIPEDIA

logger = logging.getLogger("flair")
logger.setLevel(level="INFO")


@dataclass
class ExperimentConfiguration:
    batch_size: int
    learning_rate: float
    epoch: int
    context_size: int
    seed: int
    base_model: str
    base_model_short: str
    task: str
    layers: str = "-1"
    subtoken_pooling: str = "first"
    use_crf: bool = False
    use_tensorboard: bool = True


def run_experiment(experiment_configuration: ExperimentConfiguration) -> str:
    set_seed(experiment_configuration.seed)

    # Possible task names:
    # - ner/conll03_de_revised
    # - ner/germeval14
    # - ner/germeval14_no_wiki
    # - sentiment/germeval18_coarse
    # - sentiment/germeval18_fine

    label_type = None

    if experiment_configuration.task.startswith("ner"):
        label_type = "ner"

        corpus = None

        if experiment_configuration.task == "ner/germeval14":
            corpus = NER_GERMAN_GERMEVAL()
        elif experiment_configuration.task == "ner/germeval14_no_wiki":
            corpus = NER_GERMEVAL_2014_NO_WIKIPEDIA()
        elif experiment_configuration.task == "ner/conll03_de_revised":
            corpus = CONLL_03_GERMAN_REVISED()

        label_dictionary = corpus.make_label_dictionary(label_type=label_type)
        logger.info("Label Dictionary: {}".format(label_dictionary.get_items()))

        embeddings = TransformerWordEmbeddings(
            model=experiment_configuration.base_model,
            layers=experiment_configuration.layers,
            subtoken_pooling=experiment_configuration.subtoken_pooling,
            fine_tune=True,
            use_context=experiment_configuration.context_size,
        )

        tagger = SequenceTagger(
            hidden_size=256,
            embeddings=embeddings,
            tag_dictionary=label_dictionary,
            tag_type=label_type,
            use_crf=experiment_configuration.use_crf,
            use_rnn=False,
            reproject_embeddings=False,
        )

        trainer = ModelTrainer(tagger, corpus)

        output_path_parts = [
            "flair",
            experiment_configuration.task.split("/")[0],
            experiment_configuration.task.split("/")[1].replace("_", "-"),
            experiment_configuration.base_model_short,
            f"bs{experiment_configuration.batch_size}",
            f"e{experiment_configuration.epoch}",
            f"lr{experiment_configuration.learning_rate}",
            str(experiment_configuration.seed)
        ]

        output_path = "-".join(output_path_parts)

        plugins = []

        if experiment_configuration.use_tensorboard:
            logger.info("TensorBoard logging is enabled")

            tb_path = Path(f"{output_path}/runs")
            tb_path.mkdir(parents=True, exist_ok=True)

            plugins.append(TensorboardLogger(log_dir=str(tb_path), comment=output_path))

        trainer.fine_tune(
            output_path,
            learning_rate=experiment_configuration.learning_rate,
            mini_batch_size=experiment_configuration.batch_size,
            max_epochs=experiment_configuration.epoch,
            shuffle=True,
            embeddings_storage_mode='none',
            weight_decay=0.,
            use_final_model_for_eval=False,
            plugins=plugins,
        )

        # Finally, print model card for information
        tagger.print_model_card()

        return output_path
    elif experiment_configuration.task.startswith("sentiment"):
        label_type = "class"

        if "fine" in experiment_configuration.task:
            corpus = GERMEVAL_2018_OFFENSIVE_LANGUAGE(fine_grained_classes=True)
        elif "coarse" in experiment_configuration.task:
            corpus = GERMEVAL_2018_OFFENSIVE_LANGUAGE()

        label_dict = corpus.make_label_dictionary(label_type=label_type)

        document_embeddings = TransformerDocumentEmbeddings(experiment_configuration.base_model, fine_tune=True)

        classifier = TextClassifier(document_embeddings, label_dictionary=label_dict, label_type=label_type)

        trainer = ModelTrainer(classifier, corpus)

        output_path_parts = [
            "flair",
            experiment_configuration.task.split("/")[0],
            experiment_configuration.task.split("/")[1].replace("_", "-"),
            experiment_configuration.base_model_short,
            f"bs{experiment_configuration.batch_size}",
            f"e{experiment_configuration.epoch}",
            f"lr{experiment_configuration.learning_rate}",
            str(experiment_configuration.seed)
        ]

        output_path = "-".join(output_path_parts)

        plugins = []

        if experiment_configuration.use_tensorboard:
            logger.info("TensorBoard logging is enabled")

            tb_path = Path(f"{output_path}/runs")
            tb_path.mkdir(parents=True, exist_ok=True)

            plugins.append(TensorboardLogger(log_dir=str(tb_path), comment=output_path))

        trainer.fine_tune(
            output_path,
            reduce_transformer_vocab=False,  # set this to False for slow version
            mini_batch_size=experiment_configuration.batch_size,
            max_epochs=experiment_configuration.epoch,
            learning_rate=experiment_configuration.learning_rate,
            main_evaluation_metric=("macro avg", "f1-score"),
            use_final_model_for_eval=False,
        )

        # Finally, print model card for information
        classifier.print_model_card()

        return output_path
