"""
Module defining the ChordPopulator class.

Authors: Joshua Scheck
Version: 2022-11-13
"""
from argparse import ArgumentParser
from csv import DictReader
import hashlib
import pickle
import socket
from typing import Any, List, NamedTuple


# Name of the CSV column containing the player identifier
COL_PLAYER_ID = "Player Id"
# Name of the CSV column containing the year
COL_YEAR = "Year"
# TCP receive buffer size
TCP_BUFFER_SIZE = 4096
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

    def _hash(self, value: str) -> str:
        """
        Generates a hash value for the specified node address.

        Args:
            address (str): The value to hash.

        Returns:
            str: The resulting hashed value.
        """
        encoded_val = value.encode("utf-8")
        hash_val = hashlib.sha1(encoded_val).hexdigest()
        return hash_val

    def _call_rpc(
        self, port: int, method_name: str, arg1: Any=None, arg2: Any=None
    ) -> Any:
        """
        Invokes the specified remote procedure call (RPC) with the supplied
        parameters. 

        Args
            port (int): The port of the destination Node.
            method_name (str): Name of the rpc method to invoke.
            arg1 (Any): 1st positional argument to supply to the rpc call.
            arg2 (Any): 2nd positional argument to supply to the rpc call.

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
            msg_bits = pickle.dumps((method_name, arg1, arg2))
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
            try:
                # Identify the Node for storing the key, value pair 
                # TODO - should this return successor port?
                successor_port = self._call_rpc(port, "find_successor")

                key = self._hash(f"{player_id},{year}")
                value = "" #FIXME

                # Save the key, value pair in the specified Node
                self._call_rpc(successor_port, "store_value", key, value)

            except ConnectionRefusedError as e:
                print(f"Failed to connect: {e}")
            except TimeoutError as e:
                print(f"Operation timed out: {e}") 


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