"""
Module defining the ChordQuery class.

Authors: Joshua Scheck
Version: 2022-11-13
"""
from argparse import ArgumentParser
import hashlib
import pickle
import socket
from typing import Any


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

    # FIXME - determine return type
    def query(self, key: str):
        hashed_val = self._hash(key)

        # Identify the Node for retrieving value of the specified key
        # FIXME - make find_successor return a port number?
        successor_port = self._call_rpc(self._port, "find_successor")

        return self._call_rpc(successor_port, "get_value", hashed_val)


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

    # Initializes the query Object
    querier = ChordQuery(parsed_args.port)
    # Retrieve the value mapped to the specified key from the Chord network.
    response = querier.query(parsed_args.key)

    # TODO - figure out how to format this displayed value
    print(response)