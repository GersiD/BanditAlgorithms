import JuMP
import Gurobi
using JuMP
using LinearAlgebra

# Implement simple SVM using JuMP
# Data
n = 10
dims = 2
x = randn(n, dims)
y = 2 * (x * [1, 1] .> 0) .- 1
@show x
display(x)
@show y

# Model
m = Model(Gurobi.Optimizer)
set_optimizer_attribute(m, "OutputFlag", 0)
@variable(m, w[1:dims])
@variable(m, b)
@constraint(m, [i = 1:n], y[i] * (dot(w, x[i, :]) + b) >= 1)
@objective(m, Min, sum(w .^ 2))
optimize!(m)
normal = value.(w)
offset = value(b)
obj = objective_value(m)
ρ = 1 / norm(normal, 2)

using Plots
scatter(x[:, 1], x[:, 2], color=y, label="")
# Plot a circle with radius ρ around each point
θ = range(0, 2π, length=1000)
for i in 1:n
  plot!(x[i, :][1] .+ ρ * cos.(θ), x[i, :][2] .+ ρ * sin.(θ), color=:black, linewidth=0.5, label="")
end
# Plot the decision boundary
plot!(x -> (-normal[1] * x - offset) / normal[2], xlim=(-3, 3), ylim=(-3, 3), color=:black, linewidth=2, label="SVM Decision Boundary")
savefig("svm.png")

