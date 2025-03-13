from typing import List, Optional, Union

import numpy as np
from scvi.train import TrainingPlan, TrainRunner
from scvi.dataloaders import DataSplitter

class BasicTrainingMixin:
    def train(
        self,
        max_epochs: Optional[int] = 100,
        train_size: float = 0.9,
        validation_size: float | None = None,
        batch_size: int = 128,
        early_stopping: bool = False,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """Train the model."""
        # object to make train/test/val dataloaders
        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else dict()
        data_splitter = DataSplitter(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            batch_size=batch_size,
        )
        # defines optimizers, training step, val step, logged metrics
        training_plan = TrainingPlan(
            self.module,
            **plan_kwargs
        )
        # creates Trainer, pre and post training procedures (Trainer.fit())

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )
        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            **trainer_kwargs,
        )
        return runner()