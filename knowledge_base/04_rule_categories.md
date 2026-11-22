# How to judge mined association rules

- Category 0 (invalid rule): the association is trivial, physically meaningless, or explained by normal control and weather (e.g. set-point reset with outdoor temperature, power falling with load, equipment status following staging). Such rules carry no diagnostic information.
- Category 1 (design anomaly): the association reveals a mismatch built into the system, such as oversized or undersized equipment, unsuitable design set-points, or a layout that cannot meet design conditions.
- Category 2 (operational anomaly): the association reveals wrong operation or control: sensor bias, wrong set-points, stuck or leaking valves, poorly tuned controllers, pumps or fans running more than the load requires, low delta T from excessive flow.
- Category 3 (performance degradation): the association reveals equipment losing efficiency or capacity over time, such as fouled heat exchangers or cooling-tower fill, giving higher approach temperatures, higher power and lower COP at the same conditions.
- Support says how often the pattern occurs and confidence how reliable it is; lift above 1 means the items occur together more often than by chance. A rule can have high confidence and still be invalid if it only restates normal control behaviour.
- When a rule involves outdoor dry-bulb or wet-bulb temperature, first check whether the control sequence alone explains it before calling it an anomaly.
