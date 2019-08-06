using Flux
using JuliaDB, FileIO
using JuliaDB: ML
using CuArrays
using ForwardDiff
using BSON: @save
CuArrays.culiteral_pow(::typeof(^), x::ForwardDiff.Dual{Nothing,Float32,1}, ::Val{2}) = x*x
pwd()
data_table = table(columns(JuliaDB.load("out/data.db"))..., names=(:blob_size, :dist_centroid, :dist_road, [Symbol(string("cnt", c)) for c = 4:15]..., :out))
train_table = data_table # JuliaDB.select(data_table, (1, 2, 3, 4, 16))
# FileIO.save("out/data.csv", train_table)

test_table = table(columns(JuliaDB.load("out/data_2.db"))..., names=(:blob_size, :dist_centroid, :dist_road, [Symbol(string("cnt", c)) for c = 4:15]..., :out))

sch = ML.schema(train_table, hints=Dict(
  :out => ML.Categorical,
))

test_sch = ML.schema(test_table, hints=Dict(
  :out => ML.Categorical,
))

input_sch, output_sch = ML.splitschema(sch, :out)
test_input_sch, test_output_sch = ML.splitschema(test_sch, :out)

train_input = ML.featuremat(input_sch, train_table) |> gpu
train_output = ML.featuremat(output_sch, train_table) |> gpu

test_input = ML.featuremat(test_input_sch, test_table) |> gpu
test_output = ML.featuremat(test_output_sch, test_table) |> gpu

model = Chain(
  Dense(ML.width(input_sch), 32, relu),
  Dense(32, 10),
  Dense(10, ML.width(output_sch)),
  softmax) |> gpu

loss(x, y) = Flux.crossentropy(model(x), y)

opt = Flux.ADAM()
evalcb = Flux.throttle(() -> @show(loss(train_input, train_output)), 2);

data = [(train_input, train_output)]

for i = 1:500
  Flux.train!(loss, params(model), data, opt, cb = evalcb)
end
tmp = train_input |> Array
model(train_input[:, 1])

model(test_input)

Flux.onecoldbatch(model(test_input))

accuracy(x, y) = Base.mean(Flux.onecold(model(x)) .== Flux.onecold(y))

predict = model(test_input)[1, :] .> .5
actual = test_output[1, :] .> .5

1 - sum(predict .⊻ actual) / length(actual)

loss(test_input, test_output)

probability = model(test_input).data[1, :] |> Array
JuliaDB.save(probability |> table, "out/prob.db")

@save "5819.bson" model
