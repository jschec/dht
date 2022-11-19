"""
Module defining the ChordQuery class.

Authors: Joshua Scheck
Version: 2022-11-13
"""
from argparse import ArgumentParser
import pickle
import socket
from typing import Any, Tuple


# TCP receive buffer size
TCP_BUFFER_SIZE = 4096
# Default timeout for socket connection in seconds
DEFAULT_TIMEOUT = 1.5
# Host of the ChordNode.
NODE_HOST = "localhost"


class ChordQuery:

    def __init__(self, port: int) -> None:
        """
        Constructor for the ChordQuery class.

        Args:
            port (int): Port number of an existing ChordNode.
        """
        self._port = port

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

    def retrieve_value(self, key: Tuple[str, str]) -> None:
        """
        Queries the Chord network for the value mapped to the specified key
        and displays the resulting response.

        Args:
            key (str): The sought key to query for.
        """
        # Retrieve the value mapped to the specified key
        response = self._call_rpc(self._port, "get_value", key)

        if response is None:
            print("The key could not be found")
        else:
            print(response)


if __name__ == "__main__":
    """
    Entry point of chord_query.py

    To display instructions for the CLI, execute the following:
    python3 chord_query.py --help

    To run the program, execute the following:
    python3 chord_query.py $NODE_PORT $PLAYER_ID $YEAR

    Example call:
    # Queries Chord network through Node on port 6002 for 
    # ('chriskluwe/2506178','2006') key
    `python chord_query.py 60002 chriskluwe/2506178 2006`
    """
    parser = ArgumentParser(
        description=(
            "Executes queries against a Chord network."
        )
    )
    parser.add_argument(
        "port",
        type=int,
        help="The port number of an existing ChordNode."
    )
    parser.add_argument(
        "player_id",
        type=str,
        help=(
            "The player identifier in the sought key to query the Chord"
            "network for."
        )
    )
    parser.add_argument(
        "year",
        type=str,
        help=(
            "The year in the sought key to query the Chord network for."
        )
    )

    parsed_args = parser.parse_args()

    # Initializes the query Object
    querier = ChordQuery(parsed_args.port)
    # Retrieve the value mapped to the specified key from the Chord network.
    response = querier.retrieve_value(
        (parsed_args.player_id, parsed_args.year)
    )