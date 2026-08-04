"""
Type stubs for optional `alpaca-trade-api` (install separately; conflicts with
some yfinance/websockets versions). Used so Pylance/Pyright resolve imports
without requiring the package in every environment.
"""
from typing import Any, List, Optional

class REST:
    def __init__(
        self,
        key_id: Optional[str] = None,
        secret_key: Optional[str] = None,
        base_url: Optional[str] = None,
        api_version: str = "v2",
    ) -> None: ...
    def get_account(self) -> Any: ...
    def list_positions(self) -> List[Any]: ...
    def submit_order(self, **kwargs: Any) -> Any: ...
