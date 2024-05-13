import random
import threading
from logging import INFO
from typing import Dict, List, Optional
import flwr as fl
from flwr.server import ClientManager
from flwr.common.logger import log


class CEClientManager(ClientManager):
    """Provides a pool of available clients."""

    def __init__(self) -> None:
        self.ce_server = 0
        self.deleted = {}
        self.waiting = {}
        self.clients = {}
        self._cv = threading.Condition()

    def __len__(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        return len(self.clients)

    def num_available(self) -> int:
        """Return the number of available clients.

        Returns
        -------
        num_available : int
            The number of currently available clients.
        """
        return len(self)

    def wait_for(self, num_clients: int, timeout: int = 86400) -> bool:
        """Wait until at least `num_clients` are available.

        Blocks until the requested number of clients is available or until a
        timeout is reached. Current timeout default: 1 day.

        Parameters
        ----------
        num_clients : int
            The number of clients to wait for.
        timeout : int
            The time in seconds to wait for, defaults to 86400 (24h).

        Returns
        -------
        success : bool
        """
        with self._cv:
            return self._cv.wait_for(
                lambda: len(self.clients) >= num_clients, timeout=timeout
            )

    def register(self, client) -> bool:
        """Register Flower ClientProxy instance.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy

        Returns
        -------
        success : bool
            Indicating if registration was successful. False if ClientProxy is
            already registered or can not be registered for any reason.
        """
        if client.cid in self.clients:
            return False
        self.clients[client.cid] = client
        with self._cv:
            self._cv.notify_all()

        return True

    def unregister(self, client) -> None:
        """Unregister Flower ClientProxy instance.

        This method is idempotent.

        Parameters
        ----------
        client : flwr.server.client_proxy.ClientProxy
        """
        if client.cid in self.clients:
            del self.clients[client.cid]
            with self._cv:
                self._cv.notify_all()
                
    def register_ce_server(self, client) -> None:
        self.ce_server = client
        if client.cid in self.clients:
            del self.clients[client.cid]
            with self._cv:
                self._cv.notify_all()
    
    def reregister(self, client) -> None:
        if client.cid in self.waiting:
            self.clients[client.cid] = client
            del self.waiting[client.cid]
            with self._cv:
                self._cv.notify_all()
        return [self.clients[cid] for cid in self.clients]
                
    def eliminate(self, client) -> None:
        if client.cid in self.clients:
            self.deleted[client.cid] = client
            del self.clients[client.cid]
            with self._cv:
                self._cv.notify_all()
        return [self.clients[cid] for cid in self.clients]
                
    def set_aside(self, client) -> None:
        if client.cid in self.clients:
            self.waiting[client.cid] = client
            del self.clients[client.cid]
            with self._cv:
                self._cv.notify_all()
        return [self.clients[cid] for cid in self.clients]

    def all(self):
        """Return all available clients."""
        for cid in self.waiting:
            self.register(self.waiting[cid])
        for cid in self.deleted:
            self.register(self.deleted[cid])
        self.register(self.ce_server)
        return self.clients
        
    def sample(
        self,
        num_clients: int,
        min_num_clients: Optional[int] = None,
        criterion = None,
    ):
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]
        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []
        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]
        
