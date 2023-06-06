from sydra.generate_dataset import from_dict
import hydra

from omegaconf import DictConfig


@hydra.main(config_path="config", config_name="config")
def main(config: DictConfig):
    "Generate a synthetic dataset using the configuration set at config.yaml"
    from_dict(config)


if __name__ == "__main__":
    main()
