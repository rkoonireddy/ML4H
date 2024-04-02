import numpy as np
import torch

def get_drads(inputs, model):
    gradients = []
    for input in inputs:
        input.requires_grad = True
        output = model(input)

        target_label_idx = torch.argmax(output, 1).item()
        index = np.ones((output.size()[0], 1)) * target_label_idx
        index = torch.tensor(index, dtype=torch.int64)
        output = output.gather(1, index)
        model.zero_grad()
        output.backward()
        gradient = input.grad.detach().cpu().numpy()[0]
        gradients.append(gradient)
    gradients = np.array(gradients)
    return gradients


# integrated gradients
def integrated_gradients(inputs, model, baseline, steps=50):
    inputs = torch.as_tensor(inputs, dtype=torch.float32)
    baseline = torch.as_tensor(baseline, dtype=torch.float32)
    if baseline is None:
        baseline = 0 * inputs 
    # scale inputs and compute gradients
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(0, steps + 1)]
    grads = get_drads(scaled_inputs, model)
    avg_grads = np.average(grads[:-1], axis=0)
    avg_grads = np.transpose(avg_grads, (1, 2, 0))
    delta_X = (inputs - baseline).detach().squeeze(0).cpu().numpy()
    delta_X = np.transpose(delta_X, (1, 2, 0))
    integrated_grad = delta_X * avg_grads
    return integrated_grad

def random_baseline_integrated_gradients(inputs, model, steps, num_random_trials):
    inputs = np.asarray(inputs)
    all_intgrads = []
    for i in range(num_random_trials):
        integrated_grad = integrated_gradients(inputs, model, \
                                                baseline=255.0 *np.random.random(inputs.shape), steps=steps)
        all_intgrads.append(integrated_grad)
        print('Trial number: {} out of {}'.format(i+1,num_random_trials))
    avg_intgrads = np.average(np.array(all_intgrads), axis=0)
    return avg_intgrads
