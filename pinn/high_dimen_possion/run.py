from run_comp_cost import run as run_comp_cost
from run_rel_err import run as run_rel_err

# Compare the performance of the random sampling methods with/ without RAMS in terms of error using the similar sample budget .
run_rel_err()

# Compare the performance of the random sampling methods with/ without RAMS through the computational cost required to achieve similar accuracy.
run_comp_cost()