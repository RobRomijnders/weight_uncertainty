# OLD CODE
while not converged:
  # Get the loss
  x, y = sample_batch()
  loss = loss_function(x, y, w)

  #Update the parameters
  w_grad = gradient(loss, w)
  w = update(w, w_grad)

###############################################
# NEW CODE
while not converged:
  # Get the loss
  x, y = sample_batch()
  w = approximation.sample()
  loss = loss_function(x, y, w)

  # Update the approximation
  w_grad = gradient(loss, w)
  approximation = update(approximation, w_grad)
