using BlackBoxOptim
using ArrayFire
using ProgressMeter

allowslow(AFArray, false)

cd("/home/quangio/CLionProjects/cellular_automata/cmake-build-debug")
load_images_with_pattern(needle::Union{AbstractString, Regex}) = 
  map(f -> load_image(f, false) > 0, sort(filter(s -> occursin(needle, s), readdir())))

const lands = load_images_with_pattern(r"^land\.\d+\.png$")
const roads = load_images_with_pattern(r"^road\.\d+\.png$")
const policies = load_images_with_pattern(r"^policy\.\d+\.png$")

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
  rand(AFArray, simulation_dim) < prob .* policies[idx]

@inbounds @inbounds count_neighbour(state::AFArray, idx::Number = 0x1)::AFArray =
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
  predicted = resize(simulate(config, idx), 250, 250, AF_INTERP_NEAREST)
  actual = resize(lands[idx + 1], 250, 250, AF_INTERP_NEAREST)
  count_all((predicted & actual) + 0x0)[1] / count_all((predicted | actual) + 0x0)[1]
end

@fastmath @inbounds function evaluate(x::Vector)
  config = ParameterConfig(x[1], [x[2]], [x[3]], 1000, [x[4]])
  1 - evaluate(config, 1)
end

@time ret = simulate(ParameterConfig(0.00, [0.2], [1], 12, [0.00]), 1)
ArrayFire.image(ret)
save_image("test.png", ret |> AFArray{Float32})

evaluate(ParameterConfig(0.00, [0.0], [0], 12, [0.00]), 1)

res = bboptimize(evaluate; SearchRange = [(.001, .1), (.001, .1), (.001, .1), (.001, .1)], MaxTime=100.0)