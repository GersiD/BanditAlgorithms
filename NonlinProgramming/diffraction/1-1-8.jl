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

function newtons(df, df2, x0, tol, max_iter, alpha)
  # df: derivative of f
  # df2: second derivative of f
  # tol: tolerance
  # max_iter: maximum number of iterations
  # x0: initial guess
  # alpha: learning rate

  x = x0
  iter = 0
  dfx = df(x)
  while abs(dfx) > tol && iter < max_iter
    dfx = df(x)
    x = x - alpha * (dfx / df2(x))
    iter += 1
  end
  (x, f(x), iter)
end

function L2_sqrd(p, q, v, w)
  x = ((p * v) - (q * w)) / (v + w)
  return x[1]
end

p = [-1, -1]
q = [2, 3]
v = 1
w = 2
f(x) = norm(p - [x, 0]) / v + norm([x, 0] - q) / w
df(x) = ((x - p[1]) / (v * norm(p - [x, 0]))) + ((x - q[1]) / (w * norm(q - [x, 0])))
df2(x) = (df(x + 1e-8) - df(x)) / 1e-8
# use finite difference to approximate the derivative
# df(x) = (f(x + 1e-8) - f(x)) / 1e-8

constant_step = grad_decent(f, df, 0, 1e-6, 1000, 1)
discounted_step = grad_decent(f, df, 0, 1e-6, 1000, 2, 0.9)

println("Constant step size solution (x, fx, iterations): ", constant_step)
println("Discounted step size solution (x, fx, iterations): ", discounted_step)
println("L2 squared solution x: ", L2_sqrd(p, q, v, w))
# Now we compare to newtons method
newtons_solution = newtons(df, df2, 0, 1e-6, 1000, 1)
println("Newtons solution (x, fx, iterations): ", newtons_solution)

# Validate numerically that our solution is optimal
# We can do this by checking that the gradient is close to zero
# at the solution
@show df(constant_step[1])
@show df(discounted_step[1])
@show df(newtons_solution[1])
# Lets plot the points
using Plots
using Measures
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

# Now we need to find the optimal step size
α = 0.01:0.01:5
y = [grad_decent(f, df, 0, 1e-6, 1000, a)[3] for a in α]
plot(α, y, label="Constant α", bottom_margin=5mm, left_margin=15mm, legend=:topright)
y = [grad_decent(f, df, 0, 1e-6, 1000, a, 0.9)[3] for a in α]
plot!(α, y, label="Discounted α")
xlabel!("α")
ylabel!("Iterations")
title!("Iterations vs α")
savefig("1-1-8-iterations.png")

# Now we compare to newtons_solution
α = 0.01:0.1:20
y = [grad_decent(f, df, 0, 1e-6, 1000, a)[3] for a in α]
plot(α, y, label="Steepest Grad Descent", bottom_margin=5mm, left_margin=15mm, legend=:topright)
y = [newtons(df, df2, 0, 1e-6, 1000, a)[3] for a in α]
plot!(α, y, label="Newtons Method")
xlabel!("α")
ylabel!("Iterations")
title!("Iterations vs Constant α")
savefig("1-1-8-iterations-newtons.png")
