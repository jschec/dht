"""
Module defining the ChordNode, FingerEntry, ModRange, ModRangeIter 
and PortCatalog classes.

Authors: Joshua Scheck
Version: 2022-11-13
"""
from __future__ import annotations
from argparse import ArgumentParser
import hashlib
import pickle
import socket
import threading
from typing import Any, Tuple

# m-bit identifier
M = 5
# Maximum number of nodes in Chord network
NODES = 2 ** M
# Default timeout for socket connection in seconds
DEFAULT_TIMEOUT = 1.5
# Host of the ChordNode.
NODE_HOST = "localhost"
# Port of the ChordNode. 0 designates that a random port is chosen.
LISTENER_PORT = 0
# Maximum connections for listener
LISTENER_MAX_CONN = 100
# TCP receive buffer size
TCP_BUFFER_SIZE = 4096
# The range of possible port numbers from 0 to 2^16
POSSIBLE_PORTS = 2 ** 16

BASE_PORT = 60000


class ModRangeIter(object):
    """
    Iterator class for ModRange
    """
    
    def __init__(self, mr: ModRange, i: int, j: int) -> None:
        """
        Constructor for the ModRangeIter class.

        Args:
            mr (ModRange): Reference ModRange Object to iterate through.
            i (int): The start of the interval.
            j (int): The end of the interval.
        """
        self.mr = mr
        self.i = i
        self.j = j

    def __iter__(self) -> ModRangeIter:
        """
        Retrieves the terator for referenced ModRange Object.

        Returns:
            ModRangeIter: Iterator for the referenced ModRange Object.
        """
        return ModRangeIter(self.mr, self.i, self.j)

    def __next__(self) -> int:
        """
        Retrieves the next item in this iterator. 

        Raises:
            StopIteration: If there is no next item.

        Returns:
            int: The next value in this iterator.
        """
        if self.j == len(self.mr.intervals[self.i]) - 1:
            if self.i == len(self.mr.intervals) - 1:
                raise StopIteration()
            else:
                self.i += 1
                self.j = 0
        else:
            self.j += 1
        return self.mr.intervals[self.i][self.j]


class ModRange(object):
    """
    Range-like object that wraps around 0 at some divisor using modulo arithmetic.

    >>> mr = ModRange(1, 4, 100)
    >>> mr
    <mrange [1,4)%100>
    >>> 1 in mr and 2 in mr and 4 not in mr
    True
    >>> [i for i in mr]
    [1, 2, 3]
    >>> mr = ModRange(97, 2, 100)
    >>> 0 in mr and 99 in mr and 2 not in mr and 97 in mr
    True
    >>> [i for i in mr]
    [97, 98, 99, 0, 1]
    >>> [i for i in ModRange(0, 0, 5)]
    [0, 1, 2, 3, 4]
    """

    def __init__(self, start: int, stop: int, divisor: int) -> None:
        """
        The constructor for the ModRange class.

        Args:
            start (int): The specified start interval.
            stop (int): The specified stop interval. 
            divisor (int): Divisor to apply against the intervals.
        """
        self.divisor = divisor
        self.start = start % self.divisor
        self.stop = stop % self.divisor
        # we want to use ranges to make things speedy, but if it wraps around 
        # the 0 node, we have to use two
        if self.start < self.stop:
            self.intervals = (range(self.start, self.stop),)
        elif self.stop == 0:
            self.intervals = (range(self.start, self.divisor),)
        else:
            self.intervals = (range(self.start, self.divisor), range(0, self.stop))

    def __repr__(self) -> str:
        """
        Something like the interval|node charts in the paper
        """
        return ''.format(self.start, self.stop, self.divisor)

    def __contains__(self, node_id: int) -> bool:
        """
        Determines if the specified node_id is within this finger's interval.

        Returns:
            bool: True if the node_id is within the finger's interval, 
                otherwise False.
        """
        for interval in self.intervals:
            if node_id in interval:
                return True
        
        return False

    def __len__(self) -> int:
        """
        Retrives the total amount of intervals in this ModRangeIter.

        Returns:
            int: The total amount of intervals.
        """
        total = 0
        
        for interval in self.intervals:
            total += len(interval)
        
        return total

    def __iter__(self) -> ModRangeIter:
        """
        Retrieves an iterator for this Object.

        Returns:
            ModRangeIter: Iterator for this Object.
        """
        return ModRangeIter(self, 0, -1)


class FingerEntry(object):
    """
    Row in a finger table.

    >>> fe = FingerEntry(0, 1)
    >>> fe
    
    >>> fe.node = 1
    >>> fe
    
    >>> 1 in fe, 2 in fe
    (True, False)
    >>> FingerEntry(0, 2, 3), FingerEntry(0, 3, 0)
    (, )
    >>> FingerEntry(3, 1, 0), FingerEntry(3, 2, 0), FingerEntry(3, 3, 0)
    (, , )
    >>> fe = FingerEntry(3, 3, 0)
    >>> 7 in fe and 0 in fe and 2 in fe and 3 not in fe
    True
    """
    def __init__(self, n: int, k: int, node: int=None) -> None:
        """
        Constructor for the FingerEntry class.

        Args:
            n (int): The index of a Node in the Chord network.
            k (int): The index of the successor of the referenced Node.
            node (int, optional): The identifier of the node stored in the 
                entry. Defaults to None.

        Raises:
            ValueError: If the supplied entry values are invalid.
        """
        if not (0 <= n < NODES and 0 < k <= M):
            print(f"0 <= {n} < {NODES} and 0 < {k} <= {M}")
            raise ValueError('invalid finger entry values')
        
        self.start = (n + 2**(k-1)) % NODES
        self.next_start = (n + 2**k) % NODES if k < M else n
        self.interval = ModRange(self.start, self.next_start, NODES)
        self.node = node

    def __contains__(self, node_id: int) -> bool:
        """
        Determines if the specified node_id is within this finger's interval.

        Returns:
            bool: True if the node_id is within the finger's interval, 
                otherwise False.
        """
        return node_id in self.interval


class PortCatalog:
    """
    Catalogs all of the possible port numbers in an encapsulated hash map.
    """

    def _hash(self, value: Any) -> str:
        """
        Generates a hash value for the specified node address.

        Args:
            value (Any): The value to hash.

        Returns:
            str: The resulting hashed value.
        """
        encoded_val = pickle.dumps(value)
        hashed_val = hashlib.sha1(encoded_val).hexdigest()
        return hashed_val

    def determine_bucket(self, value: int, bucket_size: int = M) -> int:
        """
        Determines the bucket position of the specified integer value.

        Args:
            value (int): Integer value to reference.
            bucket_size (int, optional): The size of the bucket. Defaults to M.

        Returns:
            int: The identified bucket.
        """
        bucket_pos = value % (2 ** bucket_size)
        return bucket_pos

    def get_bucket(self, value: str, bucket_size: int = M) -> int:
        """
        Retrieves the bucket for the specified hash value.

        Args:
            value (str): The value to hash.
            bucket_size (int, optional): The size of the bucket. Defaults to M.

        Returns:
            int: The identified bucket.
        """
        hashed_val = self._hash(value)
        id = int(hashed_val, 16) % (2 ** bucket_size)
        return id

    def lookup_node(self, node_id: str) -> Tuple[str, int]:
        """
        Retrieves the address of the sought Node.

        Args:
            node_id (int): The identifier of the sought Node.

        Returns:
            Tuple[str, int]: Host and port of the Node.
        """
        return NODE_HOST, BASE_PORT + node_id 


class ChordNode(object):
    """
    Represents a Node in the Chord network.
    """

    def __init__(self, existing_port_num: int, id: int) -> None:
        """
        Constructor for the ChordNode class.

        Args:
            existing_port_num (int): The port number of an existing node, or 0
                to start a new network.
        """
        self._host: str = None
        self._port: int = None
        self._server: socket.socket = None
        self._port_catalog = PortCatalog()
        self._id = self._port_catalog.determine_bucket(id)

        self._start_server()

        self._finger = [None] + [
            FingerEntry(self._id, k) for k in range(1, M+1)
        ] # indexing starts at 1
        self._predecessor = None
        self._keys = {}

        threading.Thread(target=self._join, args=(existing_port_num,)).start()
    
    def __repr__(self) -> str:
        """
        Retrieves a string representation of this ChordNode.

        Returns:
            str: string representation of this ChordNode
        """
        keys = "{" + ','.join(self._keys.keys()) + "}"

        fingers = ''.join(
            [
                "\t\t{:>5} {:>12} {:>5} \n".format(
                    self._finger[idx].start,
                    f"[{self._finger[idx].interval.start},{self._finger[idx].interval.stop})",
                    self._finger[idx].node
                )
                for idx in range(1, M+1)
            ]
        )

        representation = (
            "\n\t\tChordNode: {} STATE\n"
            "\t\tpredecessor: {}\n"
            "\t\tkeys: {}\n"
            "\t\tfinger table:\n"
            "\t\t{:>5} {:>12} {:>5}\n"
            "{}"
        )

        return representation.format(
            self._id, self._predecessor, keys, "start", "interval", "node", 
            fingers
        )

    @property
    def predecessor(self) -> int:
        """
        Retrieves the predecessor of this ChordNode.

        Returns:
            int: Identifier of the predecessor.
        """
        return self._predecessor

    @predecessor.setter
    def predecessor(self, id: int) -> None:
        """
        Assigns a new value to the predecessor of this ChordNode.

        Args:
            id (int): Identifier of the new predecessor node.
        """
        self._predecessor = id

    @property
    def successor(self) -> int:
        """
        Retrieves the successor of this ChordNode.

        Returns:
            int: Identifier of the successor node.
        """
        return self._finger[1].node

    @successor.setter
    def successor(self, id: int) -> None:
        """
        Assigns a new value to the successor of this ChordNode.

        Args:
            id (int): Identifier of the new successor node.
        """
        self._finger[1].node = id
    
    def _call_rpc(
        self, node_id: int, method_name: str, arg1: Any=None, arg2: Any=None
    ) -> Any:
        """
        Invokes the specified remote procedure call (RPC) with the supplied
        parameters. 

        Args
            node_id (int): The identifier of the destination Node.
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
        if node_id == self._id:
            return self._dispatch_rpc(method_name, arg1, arg2)
        
        host, port = self._port_catalog.lookup_node(node_id)

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Establish connection with target server

            s.settimeout(DEFAULT_TIMEOUT)
            s.connect((host, port))
            
            # Convert message into bit stream and send to target server
            msg_bits = pickle.dumps((method_name, arg1, arg2))
            s.sendall(msg_bits)
            
            # Retrieve and unpickle data
            data = s.recv(TCP_BUFFER_SIZE)
            response = pickle.loads(data)
            return response

    def _start_server(self) -> socket.socket:
        """
        Starts a listener socket on a random port and sets the encapsulated
        host, port, and server socket values.
        """
        listener_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener_sock.bind((NODE_HOST, BASE_PORT + self._id))
        listener_sock.listen(LISTENER_MAX_CONN)
        listener_sock.setblocking(True)

        self._host, self._port = listener_sock.getsockname()
        self._server = listener_sock
        print(
            f"node={self._id} started listender at {self._host}:{self._port}"
        )

    def find_predecessor(self, node_id: int) -> str:
        """
        Retrieves the identifier of the predecessor node.

        Args:
            node_id (int): Identifier of an arbitrary node.

        Returns:
            int: The identifier of the predecessor node.
        """
        pred_id = self._id

        while node_id not in ModRange(
            pred_id + 1, self._call_rpc(pred_id, "successor") + 1, NODES
        ):
            pred_id = self._call_rpc(
                pred_id, "closest_preceding_finger", node_id
            )

        return pred_id

    def find_successor(self, node_id: str) -> str:
        """
        Requests the successor of an arbitrary node. 

        Args:
            node_id (int): Identifier of an arbitrary node.

        Returns:
            int: The identifier of the successor.
        """
        pred_id = self.find_predecessor(node_id)
        return self._call_rpc(pred_id, "successor")

    def closest_preceding_finger(self, node_id: int) -> str:
        """
        Retrieves the closest finger proceding the specified identifier.

        Args:
            node_id (int): Identifier of an arbitrary node.

        Returns:
            id: Closest finger preceding identifier.
        """
        for idx in range(M, 0, -1):
            if self._finger[idx].node in ModRange(self._id, node_id, NODES):
                return self._finger[idx].node
                
        return self._id

    def _join(self, node_port: int) -> None:
        """
        Requests for the specified node to join the chord network.

        Args:
            node_port (int): The port number of an existing node, or 0 to start
                a new network.
        """
        node_id = self._port_catalog.determine_bucket(BASE_PORT - node_port)
        print(f"{self._id}.join({node_id})")

        # Indicates joining into an existing Chord network
        if node_port != 0:
            self._call_rpc(self._id, "init_finger_table", node_id)
            self._call_rpc(self._id, "update_others")

        # Indicates that a new Chord network is being initialized
        else:
            for idx in range(1, M+1):
                self._finger[idx].node = self._id

            self.predecessor = self._id
            
            print(f"{self._id} Initialized Chord Network")

    def _init_finger_table(self, node_id: int) -> None:
        """
        Initializes the finger table of this ChordNode.

        Args:
            node_id (int): The identifier of an arbitrary node.
        """
        self.successor = self._call_rpc(
            node_id, "find_successor", self._finger[1].start
        )

        self.predecessor = self._call_rpc(self.successor, "predecessor")       
        
        self._call_rpc(self.successor, "predecessor", self._id)

        for idx in range(1, M):
            if self._finger[idx+1].start\
                in ModRange(self._id, self._finger[idx].node, NODES):
                
                self._finger[idx+1].node = self._finger[idx].node
            
            else:
                self._finger[idx+1].node = self._call_rpc(
                    node_id, "find_successor", self._finger[idx+1].start
                )

    def _update_others(self) -> None:
        """
        Update all other node that should have this node in their finger 
        tables.
        """
        # Find last node pred_id whose idx-th finger might be this node
        for idx in range(1, M+1):  
            
            pred_id = self.find_predecessor(
                (1 + self._id - 2**(idx-1) + NODES) % NODES
            )

            self._call_rpc(pred_id, "update_finger_table", self._id, idx)

    def update_finger_table(self, node_id: int, idx: int) -> str:
        """
        If node_id is idx-th finger of this node, update this node's finger 
        table with node_id.

        Args:
            node_id (int): The identifier of an arbitrary node.
            idx (int): Index in finger table.

        Returns:
            str: Representation of finger table changes.
        """
        if (self._finger[idx].start != self._finger[idx].node
                 and node_id in ModRange(
                    self._finger[idx].start, self._finger[idx].node, NODES
                )
            ):
            self._finger[idx].node = node_id
            
            self._call_rpc(
                self._predecessor, 'update_finger_table', node_id, idx
            )
            return str(self)
        else:
            return "did nothing {}".format(self)

    def _dispatch_rpc(
        self, method_name: str, arg1: Any, arg2: Any
    ) -> Any:
        """
        Handles the invocation local invocation of the requested RPC.

        Args:
            method_name (str): Name of the requested RPC.
            arg1 (Any): 1st positional argument to supply to the RPC.
            arg2 (Any): 2nd positional argument to supply to the RPC.

        Raises:
            ValueError: If the supplied method_name and arguments are not
                supported.

        Returns:
            Any: Response from RPC.
        """
        arg1_rep = str(arg1) if arg1 is not None else ""
        arg2_rep = "," + str(arg2) if arg2 is not None else ""

        rpc_rep = "{}.{}({}{})".format(
            self._id, method_name, arg1_rep, arg2_rep
        ) 

        if method_name == "predecessor" and arg1 is None:
            result = self.predecessor
        elif method_name == "predecessor" and arg1 is not None:
            self.predecessor = arg1
            result = None
        elif method_name == "successor" and arg1 is None:
            result = self.successor
        elif method_name == "successor" and arg1 is not None:
            self.successor = arg1
            result = None
        elif method_name == "find_predecessor":
            result = self.find_predecessor(arg1)
        elif method_name == "find_successor" and arg2 is None:
            result = self.find_successor(arg1)
        elif method_name == "init_finger_table":
            result = self._init_finger_table(arg1)
        elif method_name == "update_others":
            result = self._update_others() 
        elif method_name == "update_finger_table":
            result = self.update_finger_table(arg1, arg2)
        elif method_name == "update_finger_table":
            result = self.update_finger_table(arg1, arg2)
        elif method_name == "closest_preceding_finger":
            result = self.closest_preceding_finger(arg1)
        elif method_name == "store_value":
            self._keys[arg1] = arg2
            result = None
        elif method_name == "get_value":
            result = self._keys.get(arg1, None)
        else:
            raise ValueError(
                f"The specified method invocation {method_name}(arg1, arg2) is"
                "not supported."
            )

        print("\t{} --> {}".format(rpc_rep, result))
        return result

    def _handle_rpc(self, client: socket.socket) -> None:
        """
        A helper method for handling incoming RPC calls.

        Args:
            client (socket.socket): The socket for the client's connection.
        """
        rpc = client.recv(TCP_BUFFER_SIZE)

        method, arg1, arg2 = pickle.loads(rpc)

        result = self._dispatch_rpc(method, arg1, arg2)
        
        client.sendall(pickle.dumps(result))

    def run(self) -> None:
        """
        Handles requests from the client.
        """
        while True:
            client, _ = self._server.accept()
            threading.Thread(target=self._handle_rpc, args=(client,)).start()


if __name__ == "__main__":
    """
    Entry point of chord_node.py

    To display instructions for the CLI, execute the following:
    python3 chord_node.py --help

    To run the program, execute the following:
    python3 chord_node.py $NODE_PORT $NODE_ID
    """
    parser = ArgumentParser(
        description=(
            "Initialize and join nodes to a Chord network."
        )
    )
    parser.add_argument(
        "port",
        type=int,
        help=(
            "The port number of an existing node, or 0 to start a new"
            " network."
        )
    )
    parser.add_argument(
        "id",
        type=int,
        help=(
            "The identifier (offset from base port) of the node."
        )
    )

    parsed_args = parser.parse_args()

    # Initialize a Chord node
    node = ChordNode(parsed_args.port, parsed_args.id)
    # Listen for incoming connections from other nodes or queriers
    node.run()