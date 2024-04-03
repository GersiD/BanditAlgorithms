using JuMP
using LinearAlgebra
using Gurobi

function lasso(X, y, λ)
  _, p = size(X)
  m = Model(Gurobi.Optimizer)
  JuMP.set_optimizer_attribute(m, "OutputFlag", 0)
  @variable(m, β[1:p] .>= 0)
  @constraint(m, sum(β) <= 1)
  @objective(m, Min, sum((y - X * β) .* (y - X * β)))
  optimize!(m)
  return value.(β)
end

function normalize(y)
  return y / norm(y, 1)
end

s::Number = 2
X = LinearAlgebra.I(s)
y::Vector{Float64} = abs.(sin.(1:s))
λ = 1
f(x) = sum((x - y) .* (x - y))
println("Note that lasso solves the constrained ||y - Xβ||^2 st ||β||_1 < λ problem\n")
@show lasso(X, y, λ)
@show sum(lasso(X, y, λ))

println("My algorithm just normalizes y, which only works when y has all non-negative components")
n = normalize(y)
@show n
@show sum(n)


# Lets plot the runtime of Gurobi vs the normalize function on different sizes of s
using Plots
using BenchmarkTools

sizes = range(2, 5000, step=100)
gurobi_time = []
normalize_time = []
gurobi_val = []
normalize_val = []
for i in sizes
  y = abs.(sin.(1:i))
  X = LinearAlgebra.I(i)
  f(x) = sum((x - y) .* (x - y))
  push!(gurobi_time, @elapsed lasso(X, y, 1))
  push!(normalize_time, @elapsed normalize(y))
  push!(gurobi_val, f(lasso(X, y, 1)))
  push!(normalize_val, f(normalize(y)))
end

plot(sizes, gurobi_time, label="Gurobi")
plot!(sizes, normalize_time, label="Normalize")
xlabel!("Len of y")
ylabel!("Time (s)")
title!("Runtime of Gurobi vs Normalize")
savefig("lasso_runtime.png")

plot(sizes, gurobi_val, label="Gurobi")
plot!(sizes, normalize_val, label="Normalize")
xlabel!("Len of y")
ylabel!("Value of f(x) (lower is better)")
title!("Value of f(x) of Gurobi vs Normalize")
savefig("lasso_value.png")
