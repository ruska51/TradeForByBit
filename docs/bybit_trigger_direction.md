# Bybit triggerDirection reference

Bybit's derivatives API uses the `triggerDirection` parameter to describe how
conditional orders are triggered. The numeric codes are:

| Direction | Numeric value | Description |
|-----------|---------------|-------------|
| Rising    | 1             | Trigger when the reference price moves upwards to the trigger price. |
| Falling   | 2             | Trigger when the reference price moves downwards to the trigger price. |

These values are now referenced through `main.BYBIT_TRIGGER_DIRECTIONS` to avoid
hard-coded literals in the order placement helpers.
