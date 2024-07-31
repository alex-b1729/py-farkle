def dice_remaining_convert(n: int) -> int:
    """convert 0 dice remaining to 6"""
    return int((n - 1) % 6 + 1)
