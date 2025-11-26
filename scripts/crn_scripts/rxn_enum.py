import argparse
import os
import yaml
from yamo.yamol import yamol
from reactnet.netgen.producer.gen_prods import reaction_enumeration, create_reaction_xyzs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="config.yaml", help="Path to the config file."
    )
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config: dict = yaml.safe_load(file)

    # read the input
    


if __name__ == "__main__":
    main()