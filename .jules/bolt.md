## 2024-04-07 - [O(1) lookups for list membership]
**Learning:** Checking membership repeatedly against `polars.DataFrame.columns` inside loops or list comprehensions has O(N) complexity and can be slow.
**Action:** Always convert the list to a `set` first to reduce lookup complexity from O(N) to O(1).
