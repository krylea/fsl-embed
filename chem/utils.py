
from ocpmodels.tasks.task import BaseTask
from ocpmodels.trainers.energy_trainer import EnergyTrainer

from ocpmodels.common.registry import registry


class FSLTrainer(EnergyTrainer):
    def __init__(        
        self,
        task,
        model,
        dataset,
        optimizer,
        identifier,
        normalizer=None,
        timestamp_id=None,
        run_dir=None,
        is_debug=False,
        is_hpo=False,
        print_every=100,
        seed=None,
        logger="tensorboard",
        local_rank=0,
        amp=False,
        cpu=False,
        slurm={},
        noddp=False,
    ):
        super().__init__(
            task=task,
            model=model,
            dataset=dataset,
            optimizer=optimizer,
            identifier=identifier,
            normalizer=normalizer,
            timestamp_id=timestamp_id,
            run_dir=run_dir,
            is_debug=is_debug,
            is_hpo=is_hpo,
            print_every=print_every,
            seed=seed,
            logger=logger,
            local_rank=local_rank,
            amp=amp,
            cpu=cpu,
            name="is2re",
            slurm=slurm,
            noddp=noddp,
        )



@registry.register_task("finetune")
class FinetuneTask(BaseTask):
    def run(self):
        # Note that the results won't be precise on multi GPUs due to padding of extra images (although the difference should be minor)
        assert (
            self.trainer.val_loader is not None
        ), "Val dataset is required for making predictions"
        assert self.config["checkpoint"]
        self.trainer.validate(
            split="val",
            disable_tqdm=self.config.get("hide_eval_progressbar", False),
        )
