from argparse import ArgumentParser


if __name__ == "__main__":
    """
    Entry point of chord_query.py

    To display instructions for the CLI, execute the following:
    python3 chord_query.py --help

    To run the program, execute the following:
    python3 chord_query.py $NODE_PORT $KEY
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
        "key",
        type=int,
        help="The port number the forex publisher is running on."
    )

    parsed_args = parser.parse_args()