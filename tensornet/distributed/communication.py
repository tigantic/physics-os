"""
MPI-style communication patterns for distributed CFD.

This module provides communication primitives for data exchange
between processors in distributed simulations.

Author: HyperTensor Team
"""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Callable
from enum import Enum, auto
import threading
import queue
import time


class CommPattern(Enum):
    """Communication pattern types."""
    POINT_TO_POINT = auto()
    BROADCAST = auto()
    SCATTER = auto()
    GATHER = auto()
    ALL_REDUCE = auto()
    ALL_TO_ALL = auto()


class AllReduceOp(Enum):
    """All-reduce operation types."""
    SUM = auto()
    PROD = auto()
    MAX = auto()
    MIN = auto()
    AVG = auto()


@dataclass
class Message:
    """Message for inter-process communication."""
    source: int
    destination: int
    tag: int
    data: torch.Tensor
    timestamp: float = field(default_factory=time.time)


class MessageQueue:
    """Thread-safe message queue."""
    
    def __init__(self):
        self._queue = queue.Queue()
        self._lock = threading.Lock()
    
    def put(self, message: Message):
        """Add message to queue."""
        self._queue.put(message)
    
    def get(self, block: bool = True, timeout: Optional[float] = None
           ) -> Optional[Message]:
        """Get message from queue."""
        try:
            return self._queue.get(block=block, timeout=timeout)
        except queue.Empty:
            return None
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self._queue.empty()


class Communicator:
    """
    MPI-like communicator for distributed operations.
    
    Provides collective operations and point-to-point
    communication in a multi-GPU or simulated distributed
    environment.
    
    Example:
        >>> comm = Communicator(n_procs=4)
        >>> comm.initialize()
        >>> result = comm.all_reduce(local_data, AllReduceOp.SUM)
    """
    
    def __init__(self, n_procs: int = 1, rank: int = 0):
        self.n_procs = n_procs
        self.rank = rank
        
        # Message queues (for simulated communication)
        self.inboxes: Dict[int, MessageQueue] = {
            i: MessageQueue() for i in range(n_procs)
        }
        
        # Barrier synchronization
        self._barrier_count = 0
        self._barrier_lock = threading.Lock()
        self._barrier_event = threading.Event()
        
        # Request tracking
        self._pending_requests: Dict[int, Any] = {}
        self._request_counter = 0
        
        # Collective buffers
        self._collective_buffer: Dict[int, List[torch.Tensor]] = {}
    
    def initialize(self):
        """Initialize communicator."""
        # In a real MPI environment, this would call MPI_Init
        pass
    
    def finalize(self):
        """Finalize communicator."""
        # In a real MPI environment, this would call MPI_Finalize
        pass
    
    def send(self, data: torch.Tensor, dest: int, tag: int = 0):
        """
        Blocking send.
        
        Args:
            data: Data to send
            dest: Destination rank
            tag: Message tag
        """
        message = Message(
            source=self.rank,
            destination=dest,
            tag=tag,
            data=data.clone()
        )
        
        self.inboxes[dest].put(message)
    
    def recv(self, source: int, tag: int = 0,
            timeout: Optional[float] = None) -> torch.Tensor:
        """
        Blocking receive.
        
        Args:
            source: Source rank
            tag: Message tag
            timeout: Timeout in seconds
            
        Returns:
            Received data
        """
        inbox = self.inboxes[self.rank]
        
        while True:
            message = inbox.get(block=True, timeout=timeout)
            
            if message is None:
                raise TimeoutError("Receive timed out")
            
            if message.source == source and message.tag == tag:
                return message.data
            
            # Put back if not matching
            inbox.put(message)
    
    def isend(self, data: torch.Tensor, dest: int, tag: int = 0) -> int:
        """
        Non-blocking send.
        
        Args:
            data: Data to send
            dest: Destination rank
            tag: Message tag
            
        Returns:
            Request ID
        """
        request_id = self._request_counter
        self._request_counter += 1
        
        # In real implementation, this would be async
        self.send(data, dest, tag)
        
        self._pending_requests[request_id] = {
            'type': 'send',
            'complete': True,
        }
        
        return request_id
    
    def irecv(self, source: int, tag: int = 0) -> int:
        """
        Non-blocking receive.
        
        Args:
            source: Source rank
            tag: Message tag
            
        Returns:
            Request ID
        """
        request_id = self._request_counter
        self._request_counter += 1
        
        self._pending_requests[request_id] = {
            'type': 'recv',
            'source': source,
            'tag': tag,
            'complete': False,
            'data': None,
        }
        
        return request_id
    
    def wait(self, request_id: int) -> Optional[torch.Tensor]:
        """
        Wait for non-blocking operation to complete.
        
        Args:
            request_id: Request ID from isend/irecv
            
        Returns:
            Received data for irecv, None for isend
        """
        request = self._pending_requests.get(request_id)
        
        if request is None:
            return None
        
        if request['type'] == 'recv' and not request['complete']:
            data = self.recv(request['source'], request['tag'])
            request['data'] = data
            request['complete'] = True
            return data
        
        if request['type'] == 'recv':
            return request['data']
        
        return None
    
    def test(self, request_id: int) -> bool:
        """Test if request is complete."""
        request = self._pending_requests.get(request_id)
        return request is not None and request.get('complete', False)
    
    def barrier(self):
        """
        Synchronize all processes.
        
        Blocks until all processes reach this point.
        """
        with self._barrier_lock:
            self._barrier_count += 1
            
            if self._barrier_count >= self.n_procs:
                self._barrier_count = 0
                self._barrier_event.set()
        
        self._barrier_event.wait()
        self._barrier_event.clear()


def async_send(comm: Communicator, data: torch.Tensor,
              dest: int, tag: int = 0) -> int:
    """
    Convenience function for non-blocking send.
    
    Args:
        comm: Communicator
        data: Data to send
        dest: Destination rank
        tag: Message tag
        
    Returns:
        Request ID
    """
    return comm.isend(data, dest, tag)


def async_recv(comm: Communicator, source: int, tag: int = 0) -> int:
    """
    Convenience function for non-blocking receive.
    
    Args:
        comm: Communicator
        source: Source rank
        tag: Message tag
        
    Returns:
        Request ID
    """
    return comm.irecv(source, tag)


def barrier(comm: Communicator):
    """
    Synchronization barrier.
    
    Args:
        comm: Communicator
    """
    comm.barrier()


def all_reduce(comm: Communicator, data: torch.Tensor,
              op: AllReduceOp = AllReduceOp.SUM) -> torch.Tensor:
    """
    All-reduce operation across all processes.
    
    Args:
        comm: Communicator
        data: Local data
        op: Reduction operation
        
    Returns:
        Reduced result (same on all processes)
    """
    # Simulate all-reduce
    # In real implementation, use MPI_Allreduce or NCCL
    
    result = data.clone()
    
    # Collect from all ranks (simulated)
    all_data = [data.clone() for _ in range(comm.n_procs)]
    
    # Apply reduction
    if op == AllReduceOp.SUM:
        result = sum(all_data)
    elif op == AllReduceOp.PROD:
        result = all_data[0].clone()
        for d in all_data[1:]:
            result *= d
    elif op == AllReduceOp.MAX:
        result = torch.max(torch.stack(all_data), dim=0)[0]
    elif op == AllReduceOp.MIN:
        result = torch.min(torch.stack(all_data), dim=0)[0]
    elif op == AllReduceOp.AVG:
        result = sum(all_data) / len(all_data)
    
    return result


def broadcast(comm: Communicator, data: torch.Tensor, root: int = 0
             ) -> torch.Tensor:
    """
    Broadcast data from root to all processes.
    
    Args:
        comm: Communicator
        data: Data to broadcast (only used on root)
        root: Root rank
        
    Returns:
        Broadcast data on all ranks
    """
    if comm.rank == root:
        # Send to all other ranks
        for dest in range(comm.n_procs):
            if dest != root:
                comm.send(data, dest, tag=100)
        return data
    else:
        # Receive from root
        return comm.recv(root, tag=100)


def scatter(comm: Communicator, data: Optional[List[torch.Tensor]],
           root: int = 0) -> torch.Tensor:
    """
    Scatter data from root to all processes.
    
    Args:
        comm: Communicator
        data: List of tensors to scatter (only used on root)
        root: Root rank
        
    Returns:
        Local portion of scattered data
    """
    if comm.rank == root:
        if data is None or len(data) != comm.n_procs:
            raise ValueError("Root must provide data for all ranks")
        
        # Send to all other ranks
        for dest in range(comm.n_procs):
            if dest != root:
                comm.send(data[dest], dest, tag=101)
        
        return data[root]
    else:
        return comm.recv(root, tag=101)


def gather(comm: Communicator, data: torch.Tensor, root: int = 0
          ) -> Optional[List[torch.Tensor]]:
    """
    Gather data from all processes to root.
    
    Args:
        comm: Communicator
        data: Local data to send
        root: Root rank
        
    Returns:
        List of all data on root, None on other ranks
    """
    if comm.rank == root:
        gathered = [None] * comm.n_procs
        gathered[root] = data
        
        # Receive from all other ranks
        for source in range(comm.n_procs):
            if source != root:
                gathered[source] = comm.recv(source, tag=102)
        
        return gathered
    else:
        comm.send(data, root, tag=102)
        return None


class DistributedTensor:
    """
    Tensor distributed across multiple ranks.
    
    Provides operations that automatically handle
    communication for distributed computations.
    """
    
    def __init__(self, local_data: torch.Tensor, comm: Communicator,
                 distribution: str = 'block'):
        self.local_data = local_data
        self.comm = comm
        self.distribution = distribution
        
        # Compute global shape
        local_shape = list(local_data.shape)
        
        # Gather shapes from all ranks
        shapes = gather(comm, torch.tensor(local_shape), root=0)
        
        if comm.rank == 0:
            # Compute global shape (sum along first dimension)
            global_size = sum(s[0].item() for s in shapes)
            self._global_shape = (global_size,) + tuple(local_shape[1:])
        else:
            self._global_shape = None
        
        # Broadcast global shape
        if comm.rank == 0:
            shape_tensor = torch.tensor(list(self._global_shape))
        else:
            shape_tensor = torch.zeros(len(local_shape), dtype=torch.long)
        
        self._global_shape = tuple(broadcast(comm, shape_tensor).tolist())
    
    @property
    def global_shape(self) -> Tuple[int, ...]:
        """Get global tensor shape."""
        return self._global_shape
    
    @property
    def local_shape(self) -> Tuple[int, ...]:
        """Get local tensor shape."""
        return tuple(self.local_data.shape)
    
    def sum(self) -> torch.Tensor:
        """Global sum reduction."""
        local_sum = self.local_data.sum()
        return all_reduce(self.comm, local_sum, AllReduceOp.SUM)
    
    def mean(self) -> torch.Tensor:
        """Global mean reduction."""
        local_sum = self.local_data.sum()
        global_sum = all_reduce(self.comm, local_sum, AllReduceOp.SUM)
        
        n_elements = 1
        for dim in self._global_shape:
            n_elements *= dim
        
        return global_sum / n_elements
    
    def max(self) -> torch.Tensor:
        """Global max reduction."""
        local_max = self.local_data.max()
        return all_reduce(self.comm, local_max, AllReduceOp.MAX)
    
    def min(self) -> torch.Tensor:
        """Global min reduction."""
        local_min = self.local_data.min()
        return all_reduce(self.comm, local_min, AllReduceOp.MIN)
    
    def norm(self, p: int = 2) -> torch.Tensor:
        """Global norm computation."""
        local_norm_p = (self.local_data.abs() ** p).sum()
        global_norm_p = all_reduce(self.comm, local_norm_p, AllReduceOp.SUM)
        return global_norm_p ** (1.0 / p)
    
    def to_global(self) -> Optional[torch.Tensor]:
        """
        Gather to global tensor on rank 0.
        
        Returns:
            Full tensor on rank 0, None on other ranks
        """
        gathered = gather(self.comm, self.local_data, root=0)
        
        if gathered is not None:
            return torch.cat(gathered, dim=0)
        return None


def test_communication():
    """Test communication patterns."""
    print("Testing Communication Patterns...")
    
    # Test communicator creation
    print("\n  Testing Communicator...")
    comm = Communicator(n_procs=4, rank=0)
    comm.initialize()
    
    print(f"    Created communicator: {comm.n_procs} processes")
    
    # Test message queue
    print("\n  Testing MessageQueue...")
    mq = MessageQueue()
    
    msg = Message(source=0, destination=1, tag=0, data=torch.tensor([1.0, 2.0]))
    mq.put(msg)
    
    received = mq.get(timeout=1.0)
    assert received is not None
    assert torch.allclose(received.data, msg.data)
    print("    Message queue: OK")
    
    # Test all-reduce (simulated)
    print("\n  Testing all_reduce...")
    data = torch.tensor([1.0, 2.0, 3.0])
    
    result = all_reduce(comm, data, AllReduceOp.SUM)
    expected = data * comm.n_procs
    assert torch.allclose(result, expected)
    print(f"    SUM: {result.tolist()}")
    
    result = all_reduce(comm, data, AllReduceOp.AVG)
    expected = data
    assert torch.allclose(result, expected)
    print(f"    AVG: {result.tolist()}")
    
    # Test broadcast (simulated single-rank)
    print("\n  Testing broadcast...")
    data = torch.tensor([5.0, 6.0, 7.0])
    result = broadcast(comm, data, root=0)
    assert torch.allclose(result, data)
    print(f"    Broadcast from 0: {result.tolist()}")
    
    # Test DistributedTensor
    print("\n  Testing DistributedTensor...")
    local_data = torch.randn(100, 10)
    dist_tensor = DistributedTensor(local_data, comm)
    
    print(f"    Local shape: {dist_tensor.local_shape}")
    print(f"    Global shape: {dist_tensor.global_shape}")
    
    # Test reductions
    global_sum = dist_tensor.sum()
    global_max = dist_tensor.max()
    global_norm = dist_tensor.norm()
    
    print(f"    Global sum: {global_sum.item():.4f}")
    print(f"    Global max: {global_max.item():.4f}")
    print(f"    Global norm: {global_norm.item():.4f}")
    
    # Gather to global
    global_tensor = dist_tensor.to_global()
    if global_tensor is not None:
        print(f"    Gathered shape: {global_tensor.shape}")
    
    comm.finalize()
    
    print("\nCommunication Patterns: All tests passed!")


if __name__ == "__main__":
    test_communication()
