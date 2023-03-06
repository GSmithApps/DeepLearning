using Plots
using Flux
using ProgressMeter

# define a simple 2 layer neural network
model = Chain(
  Dense(2 => 16, σ),   # activation function inside layer
  Dense(16 => 2),
  softmax)


# train the model
loss(x, y) = Flux.Losses.crossentropy(model(x), y)

lr = 0.1

# create a function with a body

function generateAndReturnDomain()
  """
  Generate a domain of points and return it
  """

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

  return domainPoints

end

domainPoints = generateAndReturnDomain()

# classify them with an exclusive or
labels = [(col[1] < .5) ⊻ (col[2] < .5) ? 1 : 0 for col in eachcol(domainPoints)]

scatter(domainPoints[1,:], domainPoints[2,:], color=labels)

labelsOneHot = Flux.onehotbatch(labels, [0,1])


data = (domainPoints, labelsOneHot)

train_loss = []


@showprogress for i in 1:100
  Flux.train!(loss, Flux.params(model), Flux.DataLoader(data, batchsize=10), opt)
  push!(train_loss, loss(domainPoints, labelsOneHot))
  lr = lr * .98
  opt = ADAM(lr)
end

# plot the loss values over time
plot(train_loss,ylims = (0,maximum(train_loss)*1.1), xlabel="Iteration", ylabel="Loss", legend=false)

p_done = scatter(domainPoints[1,:], domainPoints[2,:], zcolor=model(domainPoints)[1,:], title="Trained network", legend=false)



