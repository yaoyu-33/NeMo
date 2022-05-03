from omegaconf import OmegaConf


class TestConfig:

    def test_config(self):
        conf = OmegaConf.load('conf/config.yaml')
        s = """
        defaults:
          - _self_
          - cluster: bcm  # Leave it as bcm even if using bcp. It will be ignored for bcp.
          - data_preparation: download_gpt3_pile
          - training: gpt3/5b  # Must match training_config below.
          - conversion: convert_gpt3
          - finetuning: null
          - evaluation: gpt3/evaluate_all
          - override hydra/job_logging: stdout

        hydra:
          run:
            dir: .
          output_subdir: null

        debug: False

        run_data_preparation: True
        run_training: True
        run_conversion: True
        run_finetuning: False # Finetuning only supports T5
        run_evaluation: True

        cluster_type: bcm  # bcm or bcp. If bcm, it must match - cluster above.
        bignlp_path: ???  # Path should end with bignlp-scripts
        data_dir: ${bignlp_path}/data  # Location to store and read the data.
        base_results_dir: ${bignlp_path}/results  # Location to store the results, checkpoints and logs.
        container_mounts: # List of additional paths to mount to container. They will be mounted to same path.
          - null
        container: nvcr.io/ea-bignlp/bignlp-training:22.04-py3

        wandb_api_key_file: null  # File where the w&B api key is stored. Key must be on the first line.
        nccl_topology_xml_file: null  # This file will be exported as "export NCCL_TOPO_FILE=${nccl_topology_xml_file}"

        # GPU Mapping
        numa_mapping:
          enable: True  # Set to False to disable all mapping (performance will suffer).
          mode: unique_contiguous  # One of: all, single, single_unique, unique_interleaved or unique_contiguous.
          scope: node  # Either node or socket.
          cores: all_logical  # Either all_logical or single_logical.
          balanced: True  # Whether to assing an equal number of physical cores to each process.
          min_cores: 1  # Minimum number of physical cores per process.
          max_cores: 8  # Maximum number of physical cores per process. Can be null to use all available cores.

        # Do not modify below, use the values above instead.
        data_config: ${hydra:runtime.choices.data_preparation}
        training_config: ${hydra:runtime.choices.training}
        finetuning_config: ${hydra:runtime.choices.finetuning}
        evaluation_config: ${hydra:runtime.choices.evaluation}
        conversion_config: ${hydra:runtime.choices.conversion}
        """
        expected = OmegaConf.create(s)
        assert expected == conf, f"conf/config.yaml must be set to {expected} but it currently is {conf}."
