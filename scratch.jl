using Plots

using Colors

# randomly generate some domain points in 2D as Float32
domainPoints = rand(Float32, 2, 1000)

# classify them with an exclusive or
labels = [(col[1] < .5) ⊻ (col[2] < .5) for col in eachcol(domainPoints)]

# map the labels to 1 and 0
labels = map(x -> x ? 1 : 0, labels)

# plot the points with the labels as color
scatter(domainPoints[1,:], domainPoints[2,:], color=labels)


using Flux

# define a simple 2 layer neural network
model = Chain(
  Dense(2, 16, σ),
  Dense(16, 1, σ),
)


# train the model
loss(x, y) = Flux.mse(model(x), y)

opt = ADAM(0.1)

data = [(domainPoints[:, i], labels[i]) for i in eachindex(labels)]

for i in 1:100
  Flux.train!(loss, Flux.params(model), data, opt)
end

ŷ = model(domainPoints)

scatter(domainPoints[1,:], domainPoints[2,:], markercolor=Gray.(ŷ))

ŷ
