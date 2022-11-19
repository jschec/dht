from __future__ import annotations
from argparse import ArgumentParser
import hashlib
import pickle
import socket
import threading
from typing import Any, Dict, List, Tuple

# m-bit identifier
M = hashlib.sha1().digest_size * 8
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


class ModRangeIter(object):
    """
    Iterator class for ModRange
    """
    
    def __init__(self, mr: ModRange, i: int, j: int) -> None:
        """
        Constructor for the ModRangeIter class.

        Args:
            mr (ModRange): TODO
            i (int): TODO
            j (int): TODO
        """
        self.mr = mr
        self.i = i
        self.j = j

    def __iter__(self) -> ModRangeIter:
        return ModRangeIter(self.mr, self.i, self.j)

    def __next__(self) -> int:
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
            start (int): TODO
            stop (int): TODO
            divisor (int): TODO
        """
        self.divisor = divisor
        self.start = start % self.divisor
        self.stop = stop % self.divisor
        # we want to use ranges to make things speedy, but if it wraps around the 0 node, we have to use two
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
        TODO

        Returns:
            ModRangeIter: TODO
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
            n (int): TODO
            k (int): TODO
            node (int, optional): The identifier of the node stored in the 
                entry. Defaults to None.

        Raises:
            ValueError: If the supplied entry values are invalid.
        """
        if not (0 <= n < NODES and 0 < k <= M):
            raise ValueError('invalid finger entry values')
        
        self.start = (n + 2**(k-1)) % NODES
        self.next_start = (n + 2**k) % NODES if k < M else n
        self.interval = ModRange(self.start, self.next_start, NODES)
        self.node = node

    def __repr__(self) -> str:
        """ Something like the interval|node charts in the paper """
        return ''.format(self.start, self.next_start, self.node)

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

    def __init__(self) -> None:
        """
        Constructor for the PortCatalog class.
        """
        self._not_allowed_ports = []
        self._node_map = self._generate_node_map()

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

    def port_is_allowed(self, port_num: int) -> bool:
        """
        Determines if the specified port number is allowed.

        Args:
            port_num (int): Port number to check.

        Returns:
            bool: True if the port number is allowed, otherwise False.
        """
        return port_num not in self._not_allowed_ports

    def get_bucket(self, value: str, bucket_size: int = M) -> int:
        """
        Retrieves the bucket for the specified hash value.

        Args:
            value (str): The value to hash.
            bucket_size (int): The size of the bucket. Defaults to M.

        Returns:
            int: The identified bucket.
        """
        hashed_val = self._hash(value)
        id = int(hashed_val, 16) % (2 ** bucket_size)
        return id

    def _generate_node_map(self) -> Dict[str, List[Tuple[str, int]]]:
        """
        Generates a hash table with the hash values mapped to possible node 
        addresses.

        Returns:
            Dict[str, List[Tuple[str, int]]]: The map of possible hash values
                mapped to their respective node addresses.
        """
        node_map: Dict[str, List[Tuple[str, int]]] = {}

        for port in range(1, POSSIBLE_PORTS):
            address = (NODE_HOST, port)

            hashed_val = self.get_bucket(address)

            if hashed_val in node_map:
                self._not_allowed_ports.append(port)
                print('cannot use', address, 'hash conflict', hashed_val)
            
            else:
                node_map[hashed_val] = address

        return node_map

    def lookup_node(self, node_hash: str) -> Tuple[str, int]:
        """
        Retrieves the address of the sought Node.

        Args:
            node_hash (node_hash): Hash identifier of the sought Node.

        Returns:
            Tuple[str, int]: Host and port of the Node.
        """
        return self._node_map[node_hash]


class ChordNode(object):

    def __init__(self, existing_port_num: int) -> None:
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

        self._start_server()

        self._id = self._port_catalog.get_bucket((NODE_HOST, self._port))

        self._finger = [None] + [
            FingerEntry(self._id, k) for k in range(1, M+1)
        ] # indexing starts at 1
        self._predecessor = None
        self._keys = {}

        self._join(existing_port_num)

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
        print("Calling RPC: ", node_id, method_name, arg1, arg2)

        if node_id == self._id:
            val =  self._dispatch_rpc(method_name, arg1, arg2)
            print(f"Returned {val}")
            return val
        
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
        valid_listener_sock = False

        while not valid_listener_sock:
            print("Creating server")
            listener_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            listener_sock.bind((NODE_HOST, LISTENER_PORT))
            listener_sock.listen(LISTENER_MAX_CONN)
            listener_sock.setblocking(True)

            host, port = listener_sock.getsockname()
            
            # Set port if stored in encapsulated catalog
            if self._port_catalog.port_is_allowed(port):
                print(f"Port {port} is valid")
                self._host = host
                self._port = port
                self._server = listener_sock
                valid_listener_sock = True  
            else:
                print(f"Port {port} is NOT valid")
                listener_sock.close()  
            
        node_id = self._port_catalog.get_bucket((NODE_HOST, self._port))

        print(
            f"Started listener for id={node_id} at {self._host}:{self._port}"
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
            pred_id, self._call_rpc(pred_id, "successor"), NODES
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
        for idx in range(M, 1, -1):
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
        # Indicates joining into an existing Chord network
        if node_port != 0:
            self._init_finger_table(self._id)
            self._update_others()

        # Indicates that a new Chord network is being initialized
        else:
            for idx in range(1, M):
                self._finger[idx].node = self._id

            self.predecessor = self._id

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

        for idx in range(1, M-1):
            if self._finger[idx+1].start\
                in ModRange(self._id, self._finger[idx].node, NODES):
                
                self._finger[idx+1].node = self._finger[idx].node
            
            else:
                self._finger[idx+1].node = self._call_rpc(
                    node_id, "find_successor", self._finger[idx+1].start
                )

    # TODO - clean up
    def _update_others(self) -> None:
        """
        Update all other node that should have this node in their finger tables
        """
        # print('update_others()')
        for idx in range(1, M+1):  # find last node p whose i-th finger might be this node
            
            # FIXME: bug in paper, have to add the 1 +
            pred_id = self.find_predecessor((1 + self._id - 2**(idx-1) + NODES) % NODES)

            self._call_rpc(pred_id, "update_finger_table", self._id, idx)

    # TODO - clean up
    def update_finger_table(self, s, idx: int) -> str:
        """ if s is i-th finger of n, update this node's finger table with s """
        # FIXME: don't want e.g. [1, 1) which is the whole circle
        if (self._finger[idx].start != self._finger[idx].node
                 # FIXME: bug in paper, [.start
                 and s in ModRange(self._finger[idx].start, self._finger[idx].node, NODES)):
            
            print('update_finger_table({},{}): {}[{}] = {} since {} in [{},{})'.format(
                     s, idx, self._id, idx, s, s, self._finger[idx].start, self._finger[idx].node))
            self._finger[idx].node = s
            
            print('#', self)
            
            self._call_rpc(self._predecessor, 'update_finger_table', s, idx)
            return str(self)
        else:
            return 'did nothing {}'.format(self)

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
        if method_name == "predecessor" and arg1 is None:
            return self.predecessor
        elif method_name == "predecessor" and arg1 is not None:
            self.predecessor = arg1
            return None
        elif method_name == "successor" and arg1 is None:
            print("111", self.successor)
            return self.successor
        elif method_name == "successor" and arg1 is not None:
            self.successor = arg1
            print("221")
            return None
        elif method_name == "find_predecessor":
            return self.find_predecessor(arg1)
        elif method_name == "find_successor" and arg2 is None:
            return self.find_successor(arg1)
        elif method_name == "update_finger_table":
            return self.update_finger_table(arg1, arg2)
        elif method_name == "closest_preceding_finger":
            return self.closest_preceding_finger(arg1)
        elif method_name == "store_value":
            self._keys[arg1] = arg2
            return None
        elif method_name == "get_value":
            return self._keys.get(arg1, None)
        else:
            raise ValueError(
                f"The specified method invocation {method_name}(arg1, arg2) is"
                "not supported."
            )

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
    python3 chord_node.py $NODE_PORT
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

    parsed_args = parser.parse_args()

    # Initialize a Chord node
    node = ChordNode(parsed_args.port)
    # Listen for incoming connections from other nodes or queriers
    node.run()