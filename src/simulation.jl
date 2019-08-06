using BlackBoxOptim
using ArrayFire
using ProgressMeter

allowslow(AFArray, false);pwd()
# cd("/home/quangio/CLionProjects/cellular_automata/cmake-build-debug")
# cd("data")
cd("data/processed_working")
load_images_with_pattern(needle::Union{AbstractString, Regex}) = 
  map(f -> load_image(f, false), sort(filter(s -> occursin(needle, s), readdir())))

const lands = load_images_with_pattern(r"^land\.\d+\.png$") .> 0
const roads = load_images_with_pattern(r"^road\.\d+\.png$") .> 0
const policies = load_images_with_pattern(r"^policy\.\d+\.png$") ./ 255
const water_forest = load_images_with_pattern("water_forest.png")[1] .> 0

const simulation_dim = size(lands[1])

module Util
  using ArrayFire
  function gen_count_kernel()::Vector{AFArray{Float32,2}}
    hosts = [ones(Float32, s, s) for s = 3:2:7]
    @inbounds for i = 1:length(hosts)
        s = size(hosts[i])[1] - 1
        hosts[i][2:s, 2:s] .= 0
    end
    [h |> AFArray for h = hosts]
  end
  const neighbour_counts = gen_count_kernel()
end

struct ParameterConfig
  diffusion_factor::Float32
  breed_factor::Vector{Float32}
  spread_factor::Vector{Float32}
  spread_threshold::UInt32
  road_breed_factor::Vector{Float32}
end

@fastmath function calculate_spread_probability(size::Number)::Float32 # TODO
end

@fastmath @inbounds urbanize(prob::Union{Real,AFArray}, idx::Number)::AFArray =
  (rand(AFArray, simulation_dim) < prob .* policies[idx]) & water_forest

@inbounds count_neighbour(state::AFArray, idx::Number = 0x1)::AFArray =
  convolve2(state, Util.neighbour_counts[idx], AF_CONV_DEFAULT, AF_CONV_AUTO)

const config = ParameterConfig(0.00, [0.00], [0.1], 12, [0.00])

function simulate(config::ParameterConfig, idx::Number)
  breed_masks = [fill(1 - factor, simulation_dim) |> AFArray{Float32} for factor ∈ config.breed_factor]
  road_breed_masks = [fill(1 - factor, simulation_dim) |> AFArray{Float32} for factor ∈ config.road_breed_factor]

  @inbounds land = lands[idx]
  @inbounds road = roads[idx]
  for step = 1:20
    
    # Diffusion
    diffusion = urbanize(config.diffusion_factor, idx)
    land |= diffusion
    
    # Breed
    @inbounds for i = 1:length(config.breed_factor)
      neighbour = count_neighbour(land |> AFArray{UInt32}, i)
      breed = urbanize(1 - breed_masks[i]^neighbour, idx)
      land |= breed
    end

    # Spread
    region = regions(land, AF_CONNECTIVITY_8, UInt32)
    number_of_region = maximum(region) + 1
    hist = histogram(region, number_of_region, 0, number_of_region)
    feasible_regions = findall(hist > config.spread_threshold) - 0x1 |> Array
    mask = region > 0
    for feasible ∈ feasible_regions
      mask &= feasible != region
    end
    region = region .* (1 - mask)
    spread_probability = config.spread_factor[1] # TODO
    edge = maxfilt(region > 0, 3, 3, AF_PAD_ZERO)
    spread = urbanize(edge .* spread_probability, idx)
    land |= spread
    
    # Road Breed
    @inbounds for i = 1:length(config.road_breed_factor)
      neighbour = count_neighbour(road, i)
      breed = urbanize(1 - road_breed_masks[i]^neighbour, idx)
      land |= breed
    end
  end
  return land
end


@inbounds function evaluate(config::ParameterConfig, idx::Number)
  scale = 2
  downscale(_in::AFArray) = resize(_in, simulation_dim[1] ÷ scale, simulation_dim[2] ÷ scale, AF_INTERP_NEAREST)
  predicted = simulate(config, idx)
  actual = lands[idx + 1]
  jaccard = 1 - count_all((predicted & actual) + 0x0)[1] / count_all((predicted | actual) + 0x0)[1]
  p = count_all(predicted + .0)[1]
  a = count_all(actual + .0)[1]

  smape = abs(a - p) / (a + p)

  jaccard + 80smape
end

@fastmath @inbounds function evaluate(x::Vector)
  config = ParameterConfig(x[1], [x[2]], [x[3]], 100, [x[4]])
  evaluate(config, 1)
end

@fastmath @inbounds function visualize(x::Vector, idx::Number = 1)
  config = ParameterConfig(x[1], [x[2]], [x[3]], 100, [x[4]])
  simulate(config, idx)
end

evaluate(zeros(4))
@time test = visualize([0.0, 0.0001, 0.0001, 0.0001])
save_image("test.png", test |> AFArray{Float32})

res = bboptimize(evaluate; SearchRange = map(x -> x./10, [(1e-3, 1e-2), (1e-2, 1e-1), (1e-2, 1e-1), (1e-2, 1e-1)]), MaxTime=200.0)
@show best_candidate(res)
ret = visualize(best_candidate(res)); save_image("predict.png", ret |> AFArray{Float32})

evaluate(ParameterConfig(0.00, [0.0], [0], 12, [0.00]), 1)
