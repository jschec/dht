"""
Module defining the ChordPopulator class.

Authors: Joshua Scheck
Version: 2022-11-13
"""
from argparse import ArgumentParser
from csv import DictReader
import pickle
import socket
from typing import List, NamedTuple


# Name of the CSV column containing the player identifier
COL_PLAYER_ID = "Player Id"
# Name of the CSV column containing the year
COL_YEAR = "Year"
# TCP receive buffer size
TCP_BUFFER_SIZE = 1024
# Default timeout for socket connection in seconds
DEFAULT_TIMEOUT = 1.5
# Host of the ChordNode.
NODE_HOST = "localhost"


class ChordData(NamedTuple):
    """
    Named tuple that represents the key for a given passing statistic record.
    """

    # The identifier of the player for the career passing statistic.
    player_id: str
    # The year in which the passing statistic was recorded.
    year: int


class ChordPopulator:

    def __init__(self, fpath: str) -> None:
        """
        Constructor for the ChordPopulator class.

        Args:
            fpath (str): File path of the file to parse.
        """
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

    def _upload_data(self, port: int, player_id: str, year: int) -> None:
        """
        Invokes the specified remote procedure call (RPC) with the supplied
        parameters. 

        Args
            port (int): Port number of an existing ChordNode to call an RPC
                against.
            player_id (str): The identifier of the player for the career 
                passing statistic.
            year (int): The year in which the passing statistic was recorded.

        Raises:
            ConnectionRefusedError: If the host and port combination cannot
                be connected to.
            TimeoutError: If the operation times out.

        Returns:
            Any: Response recieved from target server.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Establish connection with target server
            s.settimeout(DEFAULT_TIMEOUT)
            s.connect((NODE_HOST, port))
            
            # Convert message into bit stream and send to target server
            msg_bits = pickle.dumps(("find_successor", player_id, year)) #FIXME
            s.sendall(msg_bits)
            
            # Retrieve and unpickle data
            data = s.recv(TCP_BUFFER_SIZE)
            response = pickle.loads(data)
            return response

    def store_data(self, port: int) -> None:
        """
        Stores the specified keys in the designated Chord network.

        Args:
            port (int): Port number of an existing ChordNode.
        """
        for player_id, year in self._records:
            self._upload_data(port, player_id, year)


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
    populator = ChordPopulator(parsed_args.input_fpath)
    # Populate the Chord network
    populator.store_data(parsed_args.port)