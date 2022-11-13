from __future__ import annotations
from argparse import ArgumentParser
import pickle
import socket

import threading
from typing import Any

M = 3  # FIXME: Test environment, normally = hashlib.sha1().digest_size * 8
NODES = 2 ** M
BUF_SZ = 4096  # socket recv arg
BACKLOG = 100  # socket listen arg
TEST_BASE = 43544  # for testing use port numbers on localhost at TEST_BASE+n

# Default timeout for socket connection in seconds
DEFAULT_TIMEOUT = 1.5
# Host of the ChordNode.
NODE_HOST = "localhost"
# Port of the ChordNode. 0 designates that a random port is chosen.
LISTENER_PORT = 0
# Maximum connections for listener
LISTENER_MAX_CONN = 100
# TCP receive buffer size
TCP_BUFFER_SIZE = 1024


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

    def __next__(self):
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
        """ Something like the interval|node charts in the paper """
        return ''.format(self.start, self.stop, self.divisor)

    def __contains__(self, id) -> bool:
        """ Is the given id within this finger's interval? """
        for interval in self.intervals:
            if id in interval:
                return True
        return False

    def __len__(self) -> int:
        total = 0
        
        for interval in self.intervals:
            total += len(interval)
        
        return total

    def __iter__(self) -> ModRangeIter:
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
            node (int, optional): TODO. Defaults to None.

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

    def __contains__(self, id) -> bool:
        """ Is the given id within this finger's interval? """
        return id in self.interval


class ChordNode(object):

    def __init__(self, port_num: int) -> None:
        """
        Constructor for the ChordNode class.

        Args:
            port_num (int): The port number of an existing node, or 0 to start 
                a new network.
        """
        self._node = port_num
        self._finger = [None] + [FingerEntry(port_num, k) for k in range(1, M+1)]  # indexing starts at 1
        self._predecessor = None
        self._keys = {}
        self._server = self._start_server()

        self._join()

    @property
    def successor(self) -> int:
        """
        Retrieves the successor of this ChordNode.

        Returns:
            int: The key of the successor node.
        """
        return self._finger[1].node

    @successor.setter
    def successor(self, id: int) -> None:
        """
        Assigns a new value to the 

        Args:
            id (int): TODO
        """
        self._finger[1].node = id


    def _handle_rpc(self, client: socket.socket) -> Any:
        pass

    def _call_rpc(
        self, port: int, method_name: str, arg1: Any=None, arg2: Any=None
    ) -> Any:
        """
        Invokes the specified remote procedure call (RPC) with the supplied
        parameters. 

        Args:
            port (int): Port of the host to send the pickled message to.
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
            print(method_name, (NODE_HOST, port), arg1, arg2)
            s.settimeout(DEFAULT_TIMEOUT)
            s.connect((NODE_HOST, port))
            
            # Convert message into bit stream and send to target server
            msg_bits = pickle.dumps((method_name, arg1, arg2))
            s.sendall(msg_bits)
            
            # Retrieve and unpickle data
            data = s.recv(TCP_BUFFER_SIZE)
            response = pickle.loads(data)
            return response

    def _start_server(self) -> socket.socket:
        """
        Starts a listener socket on a random port.

        Returns:
            socket.socket: The configured listener socket.
        """
        listener_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener_sock.bind((NODE_HOST, LISTENER_PORT)) #TODO - maybe make this 0 only if port number = 0
        listener_sock.listen(LISTENER_MAX_CONN)
        listener_sock.setblocking(True)

        host, port = listener_sock.getsockname()
        
        #FIXME
        print(
            f"Started listener for ? at {host}:{port}\n"
        )

        return listener_sock

    def find_predecessor(self, node_port: int) -> int:
        """
        Retrieves the port number of the predecessor node

        Args:
            node_port (int): The port number of an existing node in the Chord
                network.

        Returns:
            int: TODO
        """
        port = self._node

        while node_port != self._node or node_port != self.successor:
            port = self.closest_preceding_finger(node_port)

        return port

    def find_successor(self, node_port: int) -> int:
        
        """ Ask this node to find id's successor = successor(predecessor(id))"""
        pred_port = self.find_predecessor(id)
        
        return self._call_rpc(pred_port, "successor")

    def closest_preceding_finger(self, id):
        """
        Retrieves the closest finger proceding the specified identifier.

        Args:
            id (_type_): _description_
        """
        pass

    def _join(self, node_port: int) -> None:
        """
        Requests for the specified node to join the chord network.

        Args:
            node_port (int): The port number of an existing node, or 0 to start
                a new network.
        """
        # Indicates joining into an existing Chord network
        if node_port != 0:
            self._init_finger_table(node_port)
            self._update_others()

        # Indicates that a new Chord network is being initialized
        else:
            for idx in range(M):
                self._finger[idx].node = self._node

            self._predecessor = self._node

    def _init_finger_table(self, node_port: int) -> None:
        """
        Initializes the finger table of this ChordNode.

        Args:
            node_port (int): The port number of an existing node.
        """
        self._finger[1].node = self._call_rpc(
            node_port, "find_successor", self._finger[1].start
        )

        self.predecessor = self._call_rpc(self.successor, "predecessor")        
        
        self._call_rpc(self.successor, "predecessor", self._node)

        for idx in range(1, M-1):
            if self._finger[idx+1].start == self._node or\
                self._finger[idx+1].start == self._finger[idx].node:
                
                self._finger[idx+1].node = self._finger[idx].node
            
            else:
                self._finger[idx+1].node = self._call_rpc(
                    node_port, "find_successor", self._finger[idx+1].start
                )

    def _update_others(self) -> None:
        """
        Update all other node that should have this node in their finger tables
        """
        # print('update_others()')
        for i in range(1, M+1):  # find last node p whose i-th finger might be this node
            # FIXME: bug in paper, have to add the 1 +
            p = self.find_predecessor((1 + self._node - 2**(i-1) + NODES) % NODES)
            self._call_rpc(p, 'update_finger_table', self._node, i)

    def _update_finger_table(self, s, i) -> str:
        """ if s is i-th finger of n, update this node's finger table with s """
        # FIXME: don't want e.g. [1, 1) which is the whole circle
        if (self._finger[i].start != self._finger[i].node
                 # FIXME: bug in paper, [.start
                 and s in ModRange(self._finger[i].start, self._finger[i].node, NODES)):
            print('update_finger_table({},{}): {}[{}] = {} since {} in [{},{})'.format(
                     s, i, self._node, i, s, s, self._finger[i].start, self._finger[i].node))
            self._finger[i].node = s
            print('#', self)
            p = self._predecessor  # get first node preceding myself
            self._call_rpc(p, 'update_finger_table', s, i)
            return str(self)
        else:
            return 'did nothing {}'.format(self)

    def _dispatch_rpc(
        self, method_name: str, arg1: Any, arg2: Any
    ) -> Any:
        pass

    def _handle_rpc(self, client: socket.socket) -> None:
        rpc = client.recv(BUF_SZ)
        
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
            "Connects with the forex publisher"
        )
    )
    parser.add_argument(
        "port",
        type=int,
        help="The port number the forex publisher is running on."
    )

    parsed_args = parser.parse_args()

    # Initialize a Chord node
    node = ChordNode(parsed_args.port)
    # Listen for incoming connections from other nodes or queriers
    node.run()