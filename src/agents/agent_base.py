from abc import ABC, abstractmethod


class AgentBase(ABC):
    @abstractmethod
    def select_action(self, state):
        pass

    @abstractmethod
    def update(self, episode):
        pass
