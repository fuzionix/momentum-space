import re
from typing import List, Tuple

class InputValidationError(Exception):
    def __init__(self, message: str):
        super().__init__(message)
        self.message = message

def validate_inputs(
    portfolio_input: List[List],
    allow_empty: bool = False
) -> Tuple[List[str], List[float]]:
    # Step 1: Extract and clean inputs
    tickers = [t.strip().upper() for t, _ in portfolio_input if t]
    weights = [w for _, w in portfolio_input if w]

    # Step 2: Handle empty inputs
    if not tickers:
        if allow_empty:
            return [], []
        raise InputValidationError("請輸入至少一個股票代碼。")

    if not weights:
        raise InputValidationError("請輸入對應的投資組合權重。")

    # Step 3: Check identical length
    if len(tickers) != len(weights):
        raise InputValidationError(
            f"股票代碼數量 ({len(tickers)}) 與權重數量 ({len(weights)}) 不一致。"
        )

    # Step 4: Regex validation for tickers
    ticker_pattern = re.compile(r"^[A-Z0-9.\-_=@]+$")
    invalid_tickers = [t for t in tickers if not ticker_pattern.match(t)]
    if invalid_tickers:
        raise InputValidationError(
            f"無效的代碼格式 - {', '.join(invalid_tickers)}"
        )

    # Step 5: Convert weights
    try:
        weights_float = [float(w) for w in weights]
    except ValueError:
        raise InputValidationError("所有權重必須為數字。")

    # Step 6: Check non-negative weights
    if any(w < 0 for w in weights_float):
        raise InputValidationError("權重不能為負數。")

    # Step 7: Check sum approximately equals 100
    total = sum(weights_float)
    if not 99.5 <= total <= 100.5:
        raise InputValidationError(f"權重總和必須為 100（當前為 {total:.2f}）。")

    # Step 8: Normalize and return
    normalized_weights = [w / 100 for w in weights_float]

    return tickers, normalized_weights