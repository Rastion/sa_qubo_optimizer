import numpy as np
import random, math
from qubots.base_optimizer import BaseOptimizer

class SAQUBOOptimizer(BaseOptimizer):
    """
    QUBO-based optimizer using simulated annealing logic.

    """

    def __init__(self, time_limit=300, num_iterations=10000, initial_temperature=10.0, cooling_rate=0.999):
        self.time_limit = time_limit
        self.num_iterations = num_iterations
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate

    def optimize(self, problem, initial_solution=None, **kwargs):

        # Get the QUBO matrix from the problem
        Q = problem.get_qubo()

        # Get the number of variables from the QUBO matrix
        max_index = max(max(i, j) for i, j in Q.keys())
        m = max_index + 1  # Since indices are 0-based

        # --- Solve the QUBO using simulated annealing ---
        # Initialize with a random binary vector of length m.
        if initial_solution is None:
            x = np.random.randint(0, 2, size=m)
        else:
            x = np.array(initial_solution)
        current_cost = self.qubo_cost(x, Q)
        best_x = x.copy()
        best_cost = current_cost

        T = self.initial_temperature
        for it in range(self.num_iterations):
            # Pick a random variable to flip.
            p = random.randint(0, m - 1)
            x_new = x.copy()
            x_new[p] = 1 - x_new[p]
            new_cost = self.qubo_cost(x_new, Q)
            #print("Cost= "+str(new_cost)+", iteration: "+str(it))
            delta = new_cost - current_cost
            if delta < 0 or random.random() < math.exp(-delta / T):
                x = x_new
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_x = x.copy()
            T *= self.cooling_rate
            if T < 1e-8:
                break
        
        
        return best_x, best_cost

    def qubo_cost(self, x, Q):
        """Compute the QUBO cost given binary vector x and QUBO dictionary Q."""
        cost = 0
        # Q is stored for p <= q. The full quadratic form is:
        # cost = sum_p Q[p,p]*x[p] + 2 * sum_{p < q} Q[p,q]*x[p]*x[q]
        for (p, q), coeff in Q.items():
            if p == q:
                cost += coeff * x[p]
            else:
                cost += 2 * coeff * x[p] * x[q]
        return cost
