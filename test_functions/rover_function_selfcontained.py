import sys
import numpy as np
sys.path.insert(1, '.')
from ebo_core.helper import ConstantOffsetFn, NormalizedInputFn
from rover_function import create_large_domain
# from rover_utils import RoverDomain, PointBSpline, ConstObstacleCost, NegGeom, AABoxes, UnionGeom, AdditiveCosts, ConstCost

def l2cost(x, point):
    return 10 * np.linalg.norm(x - point, 1)

class RoverReward:

    def __init__(self):
        domain = create_large_domain(force_start=False,
                                force_goal=False,
                                start_miss_cost=l2cost,
                                goal_miss_cost=l2cost)
        n_points = domain.traj.npoints
        raw_x_range = np.repeat(domain.s_range, n_points, axis=1)
        self.f_max = 5.0
        f = ConstantOffsetFn(domain, self.f_max)
        self.f = NormalizedInputFn(f, raw_x_range)
        xmin, xmax = self.f.get_range()
        self.xmin = [x for x in xmin]
        self.xmax = [x for x in xmax]

    @property
    def dx(self):
        return len(self.xmin)

    def __call__(self, x):
        return self.f(x)

def main():
    import sys
    f = RoverReward()
    x = [float(s) for s in sys.argv[1:]] if len(sys.argv) == 1 + f.dx else np.random.uniform(f.xmin, f.xmax)
    print('Input = {}'.format(x))
    print('Output = {}'.format(f(x)))

if __name__ == '__main__':
    main()
