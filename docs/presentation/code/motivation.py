model = Model()

model.train(data)
model.prune()

# Actually, the next line is all we care about:
prediction, uncertainty = model.predict(input)
