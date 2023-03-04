using Plots


# try to get an initial neural network
# to learn the exclusive or function

# randomly generate some domain points in 2D as Float32
domainPoints = rand(Float32, 2, 1000)


# classify them with an exclusive or
labels = [(col[1] < .5) ⊻ (col[2] < .5) for col in eachcol(domainPoints)]


# try to plot the points.
# this will break because the labels are not numbers
# we'll convert them to numbers in the next step
scatter(domainPoints[1,:], domainPoints[2,:], color=labels)

# map the labels to 1 and 0
labels = map(x -> x ? 1 : 0, labels)

# plot the points.
scatter(domainPoints[1,:], domainPoints[2,:], color=labels)


# change the color of the points
# to be a grayscale value
scatter(domainPoints[1,:], domainPoints[2,:], markercolor=Gray.(labels))

data = [(domainPoints[:, i], labels[i]) for i in eachindex(labels)]

# start the learning
using Flux

# train the model
loss(x, y) = Flux.mse(model(x), y)

# define a simple 2 layer neural network
model = Chain(
  Dense(2, 16, σ),
  Dense(16, 1, σ),
)

train_loss = []

lr = 0.1

for i in 1:100
  opt = ADAM(lr)
  Flux.train!(loss, Flux.params(model), data, opt)
  push!(train_loss, loss(domainPoints, reshape(labels,1,:)))
  lr = lr * .98
end

# plot the loss values over time
plot(train_loss,ylims = (0,maximum(train_loss)*1.1), xlabel="Iteration", ylabel="Loss", legend=false)


ŷ = model(domainPoints)

scatter(domainPoints[1,:], domainPoints[2,:], markercolor=reshape(Int.(round.(ŷ)),:))
