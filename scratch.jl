using Plots

range = LinRange(0,1,75)

domainPoints = zeros(2,length(range) ^ 2)

i = 1

for x in range
  for y in range
    domainPoints[1,i] = x
    domainPoints[2,i] = y
    i = i + 1
  end
end

domainPoints


# classify them with an exclusive or
labels = [(col[1] < .5) ⊻ (col[2] < .5) for col in eachcol(domainPoints)]

# map the labels to 1 and 0
labels = map(x -> x ? 1 : 0, labels)

# plot the points with the labels as color
scatter(domainPoints[1,:], domainPoints[2,:], color=labels)


using Flux

labels = Flux.onehotbatch(labels, [0,1])

typeof(labels)

# define a simple 2 layer neural network
model = Chain(
  Dense(2, 16, σ),
  Dense(16, 2, σ),
)


# train the model
loss(x, y) = Flux.Losses.crossentropy(model(x), y)

opt = ADAM(0.1)

size(labels)
size(domainPoints)
data = [(domainPoints[:, i], labels[:,i]) for i in eachindex(labels)]

for i in 1:100
  Flux.train!(loss, Flux.params(model), data, opt)
end

ŷ = model(domainPoints)

scatter(domainPoints[1,:], domainPoints[2,:], markercolor=Gray.(ŷ))

ŷ
