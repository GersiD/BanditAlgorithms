using LinearAlgebra


function grad_decent(f, df, x0, tol, max_iter, alpha, γ=1.0)
  # f: function to minimize
  # df: derivative of f
  # tol: tolerance
  # max_iter: maximum number of iterations
  # alpha: learning rate
  # x0: initial guess
  # γ: discount factor for step size (default 1.0)

  x = x0
  iter = 0
  α = alpha
  dfx = df(x)
  while abs(dfx) > tol && iter < max_iter
    dfx = df(x)
    x = x - α * dfx
    α *= γ # discount the step size
    iter += 1
  end
  (x, f(x), iter)
end

function L2_sqrd(p, q, v, w)
  (p[1] * v + q[1] * w) / (v + w)
end

p = [-1, -1]
q = [2, 3]
v = 1
w = 2
f(x) = norm(p - [x, 0]) / v + norm([x, 0] - q) / w
df(x) = ((x - p[1]) / (v * norm(p - [x, 0]))) + ((x - q[1]) / (w * norm(q - [x, 0])))
# use finite difference to approximate the derivative
# df(x) = (f(x + 1e-8) - f(x)) / 1e-8

constant_step = grad_decent(f, df, 0, 1e-6, 1000, 1)
discounted_step = grad_decent(f, df, 0, 1e-6, 1000, 2, 0.9)

println("Constant step size solution (x, fx, iterations): ", constant_step)
println("Discounted step size solution (x, fx, iterations): ", discounted_step)
println("L2 squared solution x: ", L2_sqrd(p, q, v, w))

# Validate numerically that our solution is optimal
# We can do this by checking that the gradient is close to zero
# at the solution
@show df(constant_step[1])
@show df(discounted_step[1])
# Lets plot the points
using Plots
x = -2:0.01:3
y = f.(x)
plot(x, y, label="f(x)")
# draw a line from p -> x -> q
plot!([p[1], constant_step[1], q[1]], [p[2], 0, q[2]], color="green", label="")
scatter!([p[1]], [p[2]], label="p")
scatter!([q[1]], [q[2]], label="q")
# draw a line at y = 0
hline!([0], color="black", label="")
scatter!([constant_step[1]], [0], label="Constant step size solution")
scatter!([discounted_step[1]], [0], label="Discounted step size solution")
xlims!(-2, 3)
ylims!(-2, 4)
xlabel!("x")
ylabel!("f(x)")
title!("Gradient Descent")
savefig("1-1-8.png")
