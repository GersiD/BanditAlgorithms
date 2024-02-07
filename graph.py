import gurobipy as gp
from gurobipy import GRB
import numpy as np
import matplotlib.pyplot as plt

class Diffraction:
    def __init__(self, p, q, v, w):
        if (v < 0 or w < 0 or v < w):
            raise Exception('v and w must be positive and v > w')
        self.p = p
        self.q = q
        self.v = v
        self.w = w
        self.dragging = False
        self.drag_vertex = None
        self.init_plot()

    def init_plot(self):
        # define an onclick function
        plt.connect('button_press_event', self.handle_click)
        plt.connect('button_release_event', self.handle_release)
        plt.connect('motion_notify_event', self.handle_motion)

    def solve_l2_sqrd(self):
        px = self.p[0]
        qx = self.q[0]
        x = px*self.v + qx*self.w
        return x / (self.v+self.w)
    
    def solve_l2(self):
        m = gp.Model("l2")
        x = m.addVar(lb=-GRB.INFINITY, name="x")
        m.setObjective((x - self.p[0])**2 + (x - self.q[0])**2, GRB.MINIMIZE)
        m.optimize()
        if m.status != GRB.Status.OPTIMAL:
            raise Exception(f'{m} failed to find the optimal solution')
        return x.x

    def handle_click(self, event):
        c = np.array([event.xdata, event.ydata])
        if np.linalg.norm(c-self.p) <= 0.2:
            self.dragging = True
            self.drag_vertex = 'p'
        elif np.linalg.norm(c-self.q) <= 0.2:
            self.dragging = True
            self.drag_vertex = 'q'

    def handle_motion(self, event):
        if self.dragging:
            if event.xdata and event.ydata:
                match self.drag_vertex:
                    case 'p':
                        self.p = np.array([event.xdata, event.ydata])
                    case 'q':
                        self.q = np.array([event.xdata, event.ydata])
            self.draw()

    def handle_release(self, _event):
        self.dragging = False
        self.drag_vertex = None

    def draw(self):
        plt.clf()
        plt.scatter(self.p[0], self.p[1], c='r', label='p')
        plt.scatter(self.q[0], self.q[1], c='b', label='q')
        x = self.solve_l2_sqrd()
        plt.scatter(x, 0, c='g', label='L2 Sqrd')
        # x = self.solve_l2()
        # plt.scatter(x, 0, c='y', label='L2')
        # draw x axis
        plt.axvline(x=0, color='black', lw=1)
        # draw y axis
        plt.axhline(y=0, color='black', lw=1)
        # shade the fast area below x axis (q's area) 
        plt.fill_between([-10, 10], -10, 0, color='green', alpha=0.3)
        # shade the slow area above x axis (p's area)
        plt.fill_between([-10, 10], 0, 10, color='red', alpha=0.3)
        plt.xlim(-10, 10)
        plt.ylim(-10, 10)
        plt.legend()
        plt.show()


p = np.array([1,2])
q = np.array([4,-2])
v = 10
w = 2
d = Diffraction(p, q, v, w)
d.draw()
