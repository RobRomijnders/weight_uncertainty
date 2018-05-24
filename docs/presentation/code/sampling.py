def make_prediction(input):
    for param_vec in param_vecs:
        yield model.get_output(input, param_vec)
prediction = np.mean(make_prediction(input))
