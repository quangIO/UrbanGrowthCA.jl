using BlackBoxOptim
using ArrayFire
using ProgressMeter

allowslow(AFArray, false)

module Store
    const simulation_dim = (10, 10)
end


module Util
    using ArrayFire
    function gen_count_kernel()::Vector{AFArray{Float32,2}}
        hosts = [ones(Float32, s, s) for s = 3:2:7]
        for i = 1:length(hosts)
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

    road_breed_factor::Array{Float32}
end



function calculate_spread_probability(size::UInt32)::Float32
end

urbanize(prob::Union{Real, AFArray}, order::Number = 1)::AFArray =
    rand(AFArray, Store.simulation_dim) < prob

count_neighbour(state::AFArray, idx::Number = 0x1)::AFArray =
    convolve2(state, Util.neighbour_counts[idx], AF_CONV_DEFAULT, AF_CONV_AUTO)

const config = ParameterConfig(0.01, [0.01], [0.01], 100, [0.00])

function simulation(config::ParameterConfig, land₀::AFArray, road₀::AFArray)
    breed_masks = [fill(1 - breed_factor, Store.simulation_dim) |> AFArray{Float32} for breed_factor ∈ config.breed_factor]
    land = land₀
    road = road₀
    for step = 1:20
        diffusion = urbanize(config.diffusion_factor)

        for i ∈ 1:length(config.breed_factor)
            neighbour = count_neighbour(land |> AFArray{UInt32}, i)
            breed = urbanize(1 - breed_masks[i]^neighbour)
        end

        region = regions(land, AF_CONNECTIVITY_8, UInt32)
        number_of_region = maximum(region) + 1
        hist = histogram(region, number_of_region, 0, number_of_region)
        feasible_regions = findall(hist > config.spread_threshold) - 0x1 |> Array
        mask = region > 0
        for feasible ∈ feasible_regions
            mask &= feasible != region
        end

        region = region .* (1 - mask)
        # TODO

        for i ∈ 1:length(config.breed_factor)
            neighbour = count_neighbour(land, i)
            breed = urbanize(1 - breed_masks[i]^neighbour)
        end

        for i ∈ 1:length(config.road_breed_factor)
            neighbour = count_neighbour(road, i)
            breed = urbanize(1 - breed_masks[i]^neighbour)
        end
    end
end
