import torch
import torch.nn as nn
import torch.optim as optim
from gan_sinewave import Generator
import matplotlib.pyplot as plt

generator = Generator()

model_state_dict = torch.load('model_save.pt')
generator.load_state_dict(model_state_dict)

new_generated_samples = torch.randn((100, 2))
final_generated_samples = generator(new_generated_samples)
final_generated_samples = final_generated_samples.detach()
plt.figure(1)
plt.plot(final_generated_samples[:, 0], final_generated_samples[:, 1], 'o')
plt.title('Generated samples')
plt.show()



