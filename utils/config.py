import argparse

def get_configuration():
    # argument parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix',
                    help='label to add as a prefix to name;',
                    type=str,
                    required=True)
    parser.add_argument('--suffix',
                    help='label to add as a suffix to name;',
                    type=str,
                    required=True)
    parser.add_argument('--testid',
                    help='test user ID for ICHAR;',
                    type=str,
                    required=True)
    parser.add_argument('--dataset',
                    help='dataset name;',
                    type=str,
                    required=True)   
    args = parser.parse_args()

    fit_client_id = list(range(10))
    fit_client_id.remove(int(args.testid))
    validation_id = (int(args.testid) + 9) % 10  # validation id = one less id from test id
    fit_client_id.remove(int(validation_id))

    assert args.dataset in ["FEMNIST", "ICHAR", "ICSR"]
    if args.dataset == "FEMNIST":
        config = {
            "name": f"{args.prefix}_femnist_{args.suffix}", # This field should exist.
            "num_clients": 400, # number of total clients
            "fit_clients": 2,
            "eval_clients": 400,
            "evaluate_every": 20,
            "client_resources": {"num_gpus": 1/8}, # 1/n means n clients are assigned to 1 physical gpu. Too large n may cause gpu oom error.
            "dataset_name": "FEMNIST",
            "num_rounds": 2000,
            "num_classes": 62,
            "ray_config": {
                "include_dashboard": False,
            },
            "fit_config": {
                "rnd": -1, # should be filled in main function.
                "meta_learn_epochs": 5,
                "meta_update_epochs": 1,
                "batch_size": 8,
                "alpha": 0.002,
                "beta": 0.002,
            },
            "eval_config": {
                "rnd": -1, # should be filled in main function.
                "adaptation_epochs": 5,
                "alpha": 0.002,
            },
            "strategy_config": {
                "num_per_cond_per_cycle": 4,
                "num_partitions_for_multi_cond_task": 4,
                "multi_cond_multiplier": 4,
                "available_fit_client_id": list(range(10)),
                "available_eval_client_id": list(range(400)),
            },
            "args_suffix": str(args.suffix),
        }
    elif args.dataset == "ICHAR":
        config = {
            "name": f"{args.prefix}_ichar_{args.testid}_{args.suffix}", # This field should exist.
            "num_clients": 10, # number of total clients
            "fit_clients": 3,
            "eval_clients": 10,
            "evaluate_every": 4,
            "client_resources": {"num_gpus": 1/3}, # 1/n means n clients are assigned to 1 physical gpu. Too large n may cause gpu oom error.
            "dataset_name": "ICHAR",
            "num_rounds": 500,
            "num_classes": 9,
            "ray_config": {
                "include_dashboard": False,
            },
            "fit_config": {
                "rnd": -1, # should be filled in main function.
                "meta_learn_epochs": 1,
                "meta_update_epochs": 1,
                "batch_size": 64,
                "alpha": 0.005,
                "beta": 0.0001,
            },
            "eval_config": {
                "rnd": -1, # should be filled in main function.
                "adaptation_epochs": 5,
                "alpha": 0.005,
            },
            "strategy_config": {
                "num_per_cond_per_cycle": 2,
                "num_partitions_for_multi_cond_task": 3,
                "multi_cond_multiplier": 2,
                "available_fit_client_id": fit_client_id,
                "available_eval_client_id": list(range(10)),
            },
            "args_suffix": str(args.suffix),
        }
    elif args.dataset == "ICSR":
        config = {
            "name": f"{args.prefix}_icsr_{args.testid}_{args.suffix}", # This field should exist.
            "num_clients": 10, # number of total clients
            "fit_clients": 3,
            "eval_clients": 10,
            "evaluate_every": 4,
            "client_resources": {"num_gpus": 1/3}, # 1/n means n clients are assigned to 1 physical gpu. Too large n may cause gpu oom error.
            "dataset_name": "ICSR",
            "num_rounds": 500,
            "num_classes": 14,
            "ray_config": {
                "include_dashboard": False,
            },
            "fit_config": {
                "rnd": -1, # should be filled in main function.
                "meta_learn_epochs": 1,
                "meta_update_epochs": 1,
                "batch_size": 64,
                "alpha": 0.005,
                "beta": 0.0001,
            },
            "eval_config": {
                "rnd": -1, # should be filled in main function.
                "adaptation_epochs": 5,
                "alpha": 0.005,
            },
            "strategy_config": {
                "num_per_cond_per_cycle": 2,
                "num_partitions_for_multi_cond_task": 3,
                "multi_cond_multiplier": 2,
                "available_fit_client_id": fit_client_id,
                "available_eval_client_id": list(range(10)),
            },
            "args_suffix": str(args.suffix),
        }

    return config

