using Plots
using Flux


# define a simple 2 layer neural network
model = Chain(
  Dense(2 => 3, tanh),   # activation function inside layer
  BatchNorm(3),
  Dense(3 => 2),
softmax)


# train the model
loss(x, y) = Flux.Losses.crossentropy(model(x), y)

opt = ADAM(0.1)

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

labelsOneHot = Flux.onehotbatch(labels, [0,1])


data = (domainPoints, labels)

# load the data
loadedData = Flux.DataLoader(data, batchsize=10)

first(loadedData)[2]

for i in loadedData[1:10]
  
end