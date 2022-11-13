from argparse import ArgumentParser


if __name__ == "__main__":
    """
    Entry point of chord_populate.py

    To display instructions for the CLI, execute the following:
    python3 chord_populate.py --help

    To run the program, execute the following:
    python3 chord_populate.py $NODE_PORT $IN_FPATH
    """
    parser = ArgumentParser(
        description=(
            "Connects with the forex publisher"
        )
    )
    parser.add_argument(
        "port",
        type=int,
        help="The port number the forex publisher is running on."
    )
    parser.add_argument(
        "input_fpath",
        type=str,
        help="Absolute or relative file path of the input file."
    )
   
    parsed_args = parser.parse_args()