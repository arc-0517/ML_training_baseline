import os
import copy
import json
import argparse
import datetime


class TrainConfig(object):
    def __init__(self, args: argparse.Namespace = None, **kwargs):

        if isinstance(args, dict):
            attrs = args
        elif isinstance(args, argparse.Namespace):
            attrs = copy.deepcopy(vars(args))
        else:
            attrs = dict()

        if kwargs:
            attrs.update(kwargs)
        for k, v in attrs.items():
            setattr(self, k, v)

        if not hasattr(self, 'hash'):
            self.hash = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    @classmethod
    def parse_arguments(cls) -> argparse.Namespace:
        """Create a configuration object from command line arguments."""
        parents = [
            cls.base_parser(),
            cls.data_parser(),
            cls.modeling_parser()
        ]

        parser = argparse.ArgumentParser(add_help=True, parents=parents, fromfile_prefix_chars='@')
        parser.convert_arg_line_to_args = cls.convert_arg_line_to_args

        config = cls()
        parser.parse_args(namespace=config)

        return config

    @classmethod
    def from_json(cls, json_path: str):
        """Create a configuration object from a .json file."""
        with open(json_path, 'r') as f:
            configs = json.load(f)

        return cls(args=configs)

    def save(self, path: str = None):
        """Save configurations to a .json file."""
        if path is None:
            path = os.path.join(self.checkpoint_dir, 'configs.json')
        os.makedirs(os.path.dirname(path), exist_ok=True)

        attrs = copy.deepcopy(vars(self))
        attrs['checkpoint_dir'] = self.checkpoint_dir

        with open(path, 'w') as f:
            json.dump(attrs, f, indent=2)

    @property
    def checkpoint_dir(self) -> str:
        # TODO: add if needed
        ckpt = os.path.join(
            self.checkpoint_root,
            f'target+{self.target}',
            f'scaler+{self.scaler}',
            f'model+{self.model_names}',
            self.hash  # ...
        )

        os.makedirs(ckpt, exist_ok=True)
        return ckpt

    @staticmethod
    def convert_arg_line_to_args(arg_line):
        for arg in arg_line.split():
            if not arg.strip():
                continue
            yield arg

    @staticmethod
    def base_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser("Base", add_help=False)
        parser.add_argument('--checkpoint_root', type=str, default='./save_results')
        parser.add_argument('--random_state', type=int, default=0)
        parser.add_argument('--verbose', type=bool, default=True)
        parser.add_argument('--confusion_matrix', type=bool, default=True)
        return parser

    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing data-related arguments."""
        parser = argparse.ArgumentParser("Data", add_help=False)
        parser.add_argument('--data_dir', type=str, default='./data')
        parser.add_argument('--target', type=str, default='quality')
        parser.add_argument('--train_ratio', type=float, default=0.6)

        return parser

    @staticmethod
    def modeling_parser() -> argparse.ArgumentParser:

        parser = argparse.ArgumentParser("Modeling", add_help=False)
        parser.add_argument('--model_names', type=list, default=['all'])
        parser.add_argument('--scaler', type=str, default="standard", choices=['standard', 'minmax'])

        return parser