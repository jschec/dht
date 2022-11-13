"""
Module defining the ChordPopulator class.

Authors: Joshua Scheck
Version: 2022-11-13
"""
from argparse import ArgumentParser
from csv import DictReader
from typing import List, NamedTuple


# Name of the CSV column containing the player identifier
COL_PLAYER_ID = "Player Id"
# Name of the CSV column containing the year
COL_YEAR = "Year"


class ChordData(NamedTuple):
    player_id: str
    year: int


class ChordPopulator:

    def __init__(self, port: int, fpath: str) -> None:
        """
        Constructor for the ChordPopulator class.

        Args:
            port (int): Port number of an existing ChordNode.
            fpath (str): File path of the file to parse.
        """
        self._port = port
        self._records = self._read_csv(fpath)

    def _read_csv(self, fpath: str) -> List[ChordData]:
        """
        Parses the specified CSV and retrieve the identified records.

        Args:
            fpath (str): File path of the file to parse.

        Returns:
            List[ChordData]: Identified records.
        """
        records = []

        with open(fpath, newline='') as csvfile:
            reader = DictReader(csvfile)
            
            for row in reader:
                records.append(
                    ChordData(row[COL_PLAYER_ID], row[COL_YEAR])
                )

        return records


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
            "Populates the Chord network with data."
        )
    )
    parser.add_argument(
        "port",
        type=int,
        help="The port number of an existing ChordNode."
    )
    parser.add_argument(
        "input_fpath",
        type=str,
        help="Absolute or relative file path of the input file."
    )
   
    parsed_args = parser.parse_args()

    # Construct populator to read input file and extract its records
    populator = ChordPopulator(parsed_args.port, parsed_args.input_fpath)
    # Populate the Chord network
    # TODO