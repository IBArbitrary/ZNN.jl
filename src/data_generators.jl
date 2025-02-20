function GenerateLineData(rng::AbstractRNG, size::Int;)
    x = (rand(rng, Float64, (1, size)) .- 0.5) .* 2
    y = [(x[i] >= 0) ? 1 : -1 for i ∈ 1:size]
    return (x, y)
end

function GenerateSquareDiagonalData(rng::AbstractRNG, size::Int;)
    x = rand(rng, Float64, (2, size))
    y = [(x[:, i][2] >= x[:, i][1]) ? 1 : 0 for i ∈ 1:size]
    return (x, y)
end

function quadrant(point)
    bpoint = point .< 0.0
    lookup = Dict(
        [0, 0] => 1,
        [1, 0] => 2,
        [1, 1] => 3,
        [0, 1] => 4,
    )
    return lookup[bpoint]
end

function octant(point)
    bpoint = point .< 0.0
    lookup = Dict(
        [0, 0, 0] => 1,
        [1, 0, 0] => 2,
        [1, 1, 0] => 3,
        [0, 1, 0] => 4,
        [0, 0, 1] => 5,
        [1, 0, 1] => 6,
        [1, 1, 1] => 7,
        [0, 1, 1] => 8,
    )
    return lookup[bpoint]
end

function GenerateSquareQuadrantData(rng::AbstractRNG, size::Int)
    x = rand(rng, Float64, (2, size)) * 2 .- 1
    y = [quadrant(x[:, i]) for i in 1:size]
    return (x, y)
end

function GenerateCubeOctantData(rng::AbstractRNG, size::Int)
    x = rand(rng, Float64, (3, size)) * 2 .- 1
    y = [octant(x[:, i]) for i in 1:size]
    return (x, y)
end