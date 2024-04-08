import JuMP
import Gurobi
import CSV, DataFrames
using DataFrames
using JuMP

μ = CSV.read("means.csv", DataFrame)[!, 2]
n = length(μ)
CovVec = CSV.read("covariances.csv", DataFrame)[!, 3]
C = zeros(n, n)
# Need to get n x n matrix from CovVec
# CovVec is a vector of length n^2
# CovVec[1] is C[1, 1]
# CovVec[2] is C[1, 2]
# ... CovVec[n] is C[1, n]
# CovVec[n+1] is C[2, 1]
C = reshape(CovVec, n, n)

m = Model(Gurobi.Optimizer)
set_optimizer_attribute(m, "OutputFlag", 0)
@variable(m, x[1:n] >= 0)
@constraint(m, sum(x) == 1000)
@objective(m, Min, sum(x .* (C * x)) - sum(μ .* x))
optimize!(m)
x = value.(x)
obj = objective_value(m)
# println("Optimal portfolio: ", x)
@assert sum(x) ≈ 1000
@show maximum(x)
@show argmax(x)
@show obj

