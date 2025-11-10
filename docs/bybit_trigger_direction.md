# Bybit triggerDirection reference

Bybit's derivatives API uses the `triggerDirection` parameter to describe how
conditional orders are triggered. The modern API expects explicit numeric
values:

| Direction | API value | Description |
|-----------|-----------|-------------|
| Rising    | `1`       | Trigger when the reference price moves upwards to the trigger price. |
| Falling   | `2`       | Trigger when the reference price moves downwards to the trigger price. |

These values are referenced through `main.BYBIT_TRIGGER_DIRECTIONS` to avoid
hard-coded literals in the order placement helpers and keep parity with the
official API requirements.
