def make_prediction(input):
    """
    Generates predictions using the model with different parameter vectors.

        Iterates through a set of parameter vectors and yields the model's output
        for each vector given the input.

        Args:
            input: The input data for prediction.

        Returns:
            A generator yielding the model's outputs. Each output corresponds to a
            prediction made with a different parameter vector.
    """
    for param_vec in param_vecs:
        yield model.get_output(input, param_vec)


prediction = np.mean(make_prediction(input))
