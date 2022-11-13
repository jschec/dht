from __future__ import annotations
from argparse import ArgumentParser
import pickle
import socket

import threading

M = 3  # FIXME: Test environment, normally = hashlib.sha1().digest_size * 8
NODES = 2 ** M
BUF_SZ = 4096  # socket recv arg
BACKLOG = 100  # socket listen arg
TEST_BASE = 43544  # for testing use port numbers on localhost at TEST_BASE+n


# Host of the ChordNode.
LISTENER_HOST = "localhost"
# Port of the ChordNode. 0 designates that a random port is chosen.
LISTENER_PORT = 0
# Maximum connections for listener
LISTENER_MAX_CONN = 100


class ModRangeIter(object):
    """ Iterator class for ModRange """
    
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
            ValueError: TODO
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

    def __init__(self, n) -> None:
        """
        Constructor for the ChordNode class.

        Args:
            n (TODO): TODO
        """
        self.node = n
        self.finger = [None] + [FingerEntry(n, k) for k in range(1, M+1)]  # indexing starts at 1
        self.predecessor = None
        self.keys = {}
        self._server = self._start_server()

    @property
    def successor(self):
        return self.finger[1].node

    @successor.setter
    def successor(self, id):
        self.finger[1].node = id


    def _start_server(self) -> socket.socket:
        """
        Starts a listener socket on a random port.

        Returns:
            socket.socket: The configured listener socket.
        """
        listener_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener_sock.bind((LISTENER_HOST, LISTENER_PORT))
        listener_sock.listen(LISTENER_MAX_CONN)

        host, port = listener_sock.getsockname()
        
        #FIXME
        print(
            f"Started listener for ? at {host}:{port}\n"
        )

        return listener_sock

    def find_predecessor(self, id):
        """
        Retrieves the node's 

        Args:
            id (_type_): _description_
        """
        pass

    def find_successor(self, id):
        """ Ask this node to find id's successor = successor(predecessor(id))"""
        np = self.find_predecessor(id)
        
        return self.call_rpc(np, 'successor')

    def closest_preceding_finger(self, id):
        """
        Retrieves the closest finger proceding the specified identifier.

        Args:
            id (_type_): _description_
        """
        pass

    # TODO - maybe join is for self?
    def join(self, node: ChordNode) -> None:
        """
        Requests for the specified node to join the chord network.

        Args:
            node (ChordNode): The node to join into the network.
        """
        self.predecessor = None
        self.successor = self.find_successor()

    def init_finger_table(self):
        pass

    def update_others(self) -> None:
        """ Update all other node that should have this node in their finger tables """
        # print('update_others()')
        for i in range(1, M+1):  # find last node p whose i-th finger might be this node
            # FIXME: bug in paper, have to add the 1 +
            p = self.find_predecessor((1 + self.node - 2**(i-1) + NODES) % NODES)
            self.call_rpc(p, 'update_finger_table', self.node, i)

    def update_finger_table(self, s, i) -> str:
        """ if s is i-th finger of n, update this node's finger table with s """
        # FIXME: don't want e.g. [1, 1) which is the whole circle
        if (self.finger[i].start != self.finger[i].node
                 # FIXME: bug in paper, [.start
                 and s in ModRange(self.finger[i].start, self.finger[i].node, NODES)):
            print('update_finger_table({},{}): {}[{}] = {} since {} in [{},{})'.format(
                     s, i, self.node, i, s, s, self.finger[i].start, self.finger[i].node))
            self.finger[i].node = s
            print('#', self)
            p = self.predecessor  # get first node preceding myself
            self.call_rpc(p, 'update_finger_table', s, i)
            return str(self)
        else:
            return 'did nothing {}'.format(self)


    def handle_rpc(self, client: socket.socket) -> None:
        rpc = client.recv(BUF_SZ)
        
        method, arg1, arg2 = pickle.loads(rpc)

        result = self.dispatch_rpc(method, arg1, arg2)
        
        client.sendall(pickle.dumps(result))


    def run(self):
        while True:
            client, client_addr = server.accept()
            threading.Thread(target=self.handle_rpc, args=(client,)).start()


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
        "host",
        type=str,
        help="Name of host running the forex publisher."
    )
    parser.add_argument(
        "port",
        type=int,
        help="The port number the forex publisher is running on."
    )

    parsed_args = parser.parse_args()