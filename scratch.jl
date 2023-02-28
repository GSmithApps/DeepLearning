using Plots

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
  Dense(2, 16, relu),
  Dense(16, 1, σ),
)

methods(model)
methods(+)

[1;
1]

# make a 1x2 matrix
[1 1]

# make a 2x1 matrix
typeof([1;
1])

x = [1.;
1.]

Matrix{Float64}(x)

hcat(x)

reshape(x,:,1)

reshape(x,1,:)

z = model([1,2])
z

# define the loss function
loss(x, y) = Flux.mse(model(x), y)


# define a target variable
target = labels

# make the target into a matrix
target = reshape(target, 1, length(target))


# train the model
for i in 1:1000
  Flux.train!(loss, Flux.params(model), [(domainPoints, target)], ADAM(0.1))
end

model(domainPoints[1:2,1])
target[1]
# make a y hat variable
ŷ = model(domainPoints)

ŷ

scaled = 255 .* ŷ

myColors = RGB.(scaled,scaled,scaled)

scatter!(domainPoints[1,:], domainPoints[2,:], color=myColors)

# map float to color
