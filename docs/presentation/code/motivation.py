model = Model()

model.train(data)

if application == 'embedded':
    model.prune()

# Actually, the next line is all we care about:
prediction, uncertainty = model.predict(input)
