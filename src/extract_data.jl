using ArrayFire
using Images, ImageView, ImageDraw
using NearestNeighbors
using Random

cd("data/processed_working")
try mkdir("out") catch; @info "folder already exists" end

load_images_with_pattern(needle::Union{AbstractString, Regex}) = 
  map(f -> load_image(f, false), sort(filter(s -> occursin(needle, s), readdir())))

const lands = load_images_with_pattern(r"^land\.\d+\.png$") .> 0
const roads = load_images_with_pattern(r"^road\.\d+\.png$") .> 0
const policies = load_images_with_pattern(r"^policy\.\d+\.png$") ./ 255
const water_forest = load_images_with_pattern("water_forest.png")[1] .> 0
const simulation_dim = size(lands[1])
const one_matrices = [ones(Float32, s, s) |> AFArray for s ∈ 3:2:13]

land = lands[1]
road = roads[1]
land_target = lands[2]
region = regions(land, AF_CONNECTIVITY_8, UInt32)
number_of_region = maximum(region) + 1
hist = histogram(region, number_of_region, 0, number_of_region)
feasible_regions = Array(findall(hist > 100) - 0x1)[2:end]


neighbors = convolve2(land + UInt32(0), one_matrices[5], 0x00000, 0x00000) |> Array

m00 = Float64[]
m01 = Float64[]
m10 = Float64[]

function draw_from_xy(coordinates::Array)
  tmp = (land < 0) |> Array |> Array{Gray{N0f8}};
  @inbounds for center = eachcol(coordinates) 
    draw!(tmp, CirclePointRadius(center[1], center[2], 1))
  end
  tmp
end

function real_indices(idx_array::AFArray)
  x = idx_array / simulation_dim[1]
  y = idx_array - x .* simulation_dim[1]
  [Array(x + 1) Array(y + 1)]' |> Array
end

@inbounds for feasible ∈ feasible_regions
  mask = feasible == region
  push!(m00, moments_all(mask, AF_MOMENT_M00))
  push!(m10, moments_all(mask, AF_MOMENT_M10))
  push!(m01, moments_all(mask, AF_MOMENT_M01))
end


centroids = (round.([m10./m00 m01./m00])' |> Array{Int}) # [X Y] ordering
road_xy = findall(road) - 0x1 |> real_indices

centroids_tree = KDTree(centroids |> Array{Float64})
roads_tree = KDTree(road_xy |> Array{Float64})

turned = findall((land ⊻ land_target) & !land) - 0x1 |> real_indices
not_turned_all = findall(!land_target) - 0x1 |> Array
p = size(turned)[2] / size(not_turned_all)[1]
not_turned = randsubseq(not_turned_all, p) |> AFArray |> real_indices

indices_centroid, dists_centroid = knn(centroids_tree, turned, 3, true)
indices_road, dists_road = knn(roads_tree, turned, 3, true)

indices_centroid[1]
i = 1
turned
centroids
for i = 1:size(turned)[2]
  @show turned[:, i], m00[indices_centroid[i][1]], dists_centroid[i][1]
  neighbors[1493, 264]
  return 
end
# draw!(tmp, CirclePointRadius(500, 50, 100))
save_image("out/tmp.png", turned)

