import argparse

def get_configuration():
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('-alpha',
                    help='learning rate alpha;',
                    type=float,
                    required=True)
    parser.add_argument('--meta-learn-epochs',
                    help='number of epochs for meta-learn and adaptation;',
                    type=int,
                    required=True)
    parser.add_argument('--extra-label',
                    help='label to add as a suffix to name;',
                    type=str,
                    required=True)
    args = parser.parse_args()

    config = {
        "name": f"tester3_whole_{args.extra_label}", # This field should exist.
        "num_clients": 300, # number of total clients
        "fraction_fit": 0.05, # {fraction_fit * num_clients} clients are used for training. Only number matters since dataset are split.
        "fraction_eval": 1.0, # {fraction_eval * num_clients} clients are used for testing. Only number matters since dataset are split.
        "client_resources": {"num_gpus": 1/4}, # 1/n means n clients are assigned to 1 physical gpu. Too large n may cause gpu oom error.
        "dataset_name": "femnist",
        "num_rounds": 1000,
        "ray_config": {
            "include_dashboard": True
        },
        "fit_config": {
            "rnd": -1, # should be filled in main function.
            "meta_learn_epochs": str(args.meta_learn_epochs),
            "meta_update_epochs": str(1),
            "batch_size": str(10),
            "alpha": str(args.alpha),
            "beta": str(0.010),
        },
        "eval_config": {
            "rnd": -1, # should be filled in main function.
            "adaptation_epochs": str(args.meta_learn_epochs),
            "alpha": str(args.alpha),
        },
        "args_alpha": str(args.alpha),
        "args_meta_learn_epochs": str(args.meta_learn_epochs),
        "args_extra_label": str(args.extra_label),
    }

    return config

