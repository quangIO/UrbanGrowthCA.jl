using Flux
using JuliaDB, FileIO
using JuliaDB: ML
using CuArrays
pwd()
data_table = table(columns(JuliaDB.load("out/data.db"))..., names=(:blob_size, :dist_centroid, :dist_road, [Symbol(string("cnt", c)) for c = 4:15]..., :out))
train_table = data_table # JuliaDB.select(data_table, (1, 2, 3, 4, 16))
FileIO.save("out/data.csv", train_table)

sch = ML.schema(train_table, hints=Dict(
  :out => ML.Categorical,
))

input_sch, output_sch = ML.splitschema(sch, :out)

train_input = ML.featuremat(input_sch, train_table)
train_output = ML.featuremat(output_sch, train_table)

train_input

culiteral_pow(::typeof(^), x::T, ::Val{0}) where {T<:Real} = one(x)
culiteral_pow(::typeof(^), x::T, ::Val{1}) where {T<:Real} = x
culiteral_pow(::typeof(^), x::T, ::Val{2}) where {T<:Real} = x * x
culiteral_pow(::typeof(^), x::T, ::Val{3}) where {T<:Real} = x * x * x
culiteral_pow(::typeof(^), x::T, ::Val{p}) where {T<:Real,p} = CUDAnative.pow(x, Int32(p))

model = Chain(
  Dense(ML.width(input_sch), 32, relu),
  Dense(32, 10),
  Dense(10, ML.width(output_sch)),
  softmax)

loss(x, y) = Flux.mse(model(x), y)
opt = Flux.ADAM(0.01)
evalcb = Flux.throttle(() -> @show(loss(first(data)...)), 2);

data = [(train_input, train_output)]
for i = 1:10
  Flux.train!(loss, data, opt, cb = evalcb)
end

model(train_input[:, end])