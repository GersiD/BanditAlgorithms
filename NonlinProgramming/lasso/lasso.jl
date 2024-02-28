using JuMP
using LinearAlgebra
using Gurobi

function lasso(X, y, λ)
  _, p = size(X)
  m = Model(Gurobi.Optimizer)
  JuMP.set_optimizer_attribute(m, "OutputFlag", 0)
  @variable(m, β[1:p])
  @variable(m, t[1:p] >= 0)
  @constraint(m, t .>= β)
  @constraint(m, t .>= -β)
  @objective(m, Min, sum((y - X * β) .* (y - X * β)) + λ * sum(t))
  optimize!(m)
  return value.(β)
end

s::Number = 30
X = LinearAlgebra.I(s)
y::Vector{Float64} = sin.(1:s)
λ = 1
println("Note that lasso solves the ||y - Xβ||^2 + λ||β||_1 problem\n")
@show lasso(X, y, λ)
println("\nThe solution differs from the OLS solution, which is β = X \\ y\n")
@show X \ y
