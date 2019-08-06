using Flux
using JuliaDB
using JuliaDB: ML

pwd()
train_table = table(columns(JuliaDB.load("out/data.db"))..., names=(:centroid_id, :dist_centroid, :dist_road, [Symbol(string("cnt", c)) for c = 4:15]..., :out))

sch = ML.schema(train_table, hints=Dict(
  :out => ML.Categorical,
))

input_sch, output_sch = ML.splitschema(sch, :out)

train_input = ML.featuremat(input_sch, train_table)
train_output = ML.featuremat(output_sch, train_table)

model = Chain(
  Dense(ML.width(input_sch), 32, relu),
  Dense(32, 8),
  Dense(8, ML.width(output_sch)),
  softmax)

loss(x, y) = Flux.mse(model(x), y)
opt = Flux.ADAM(0.1)
evalcb = Flux.throttle(() -> @show(loss(first(data)...)), 2);

data = [(train_input, train_output)]
for i = 1:10
  Flux.train!(loss, data, opt, cb = evalcb)
end